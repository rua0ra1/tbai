// clang-format off
#include <pinocchio/fwd.hpp>
// clang-format on

#include <vector>

#include <tbai_core/config/YamlConfig.hpp>

#include "tbai_rl/BobController.hpp"

#include <chrono>

#include <pinocchio/math/rpy.hpp>
#include <ros/ros.h>

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <urdf/model.h>
#include <kdl_parser/kdl_parser.hpp>
#include <geometry_msgs/TransformStamped.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace tbai {
namespace rl {

inline int mod(int a, int b) {
    return (a % b + b) % b;
}

BobController::BobController(const std::shared_ptr<tbai::core::StateSubscriber> &stateSubscriberPtr)
    : stateSubscriberPtr_(stateSubscriberPtr) {
    ros::NodeHandle nh;

    // Load PD
    kp_ = tbai::core::fromRosConfig<scalar_t>("bob_controller/kp");
    kd_ = tbai::core::fromRosConfig<scalar_t>("bob_controller/kd");
    jointNames_ = tbai::core::fromRosConfig<std::vector<std::string>>("joint_names");

    ik_ = getInverseKinematicsUnique();
    cpg_ = getCentralPatternGeneratorUnique();
    gridmap_ = tbai::gridmap::getGridmapInterfaceUnique();
    refVelGen_ = tbai::reference::getReferenceVelocityGeneratorUnique(nh);

    setupPinocchioModel();
    // generateSamplingPositions();

    auto modelPath = tbai::core::fromRosConfig<std::string>("bob_controller/model_path");
    ROS_INFO_STREAM("[BobController] Loading model from: " << modelPath);

    try {
        model_ = torch::jit::load(modelPath);
    } catch (const c10::Error &e) {
        std::cerr << "Could not load model from: " << modelPath << std::endl;
        throw std::runtime_error("Could not load model");
    }

    std::vector<torch::jit::IValue> stack;
    model_.get_method("set_hidden_size")(stack);
    resetHistory();

    auto legHeights = cpg_->legHeights();
    jointAngles2_ = ik_->solve(legHeights);

    standJointAngles_ = tbai::core::fromRosConfig<vector_t>("static_controller/stand_controller/joint_angles");
}

void BobController::setupPinocchioModel() {
    ros::NodeHandle nh;
    auto urdf = ros::param::param<std::string>("robot_description", "");

    pinocchio::urdf::buildModelFromXML(urdf, pinocchio::JointModelFreeFlyer(), pinocchioModel_);
    pinocchioData_ = pinocchio::Data(pinocchioModel_);
}

void BobController::visualize() {
    auto state = getBobnetState();
    stateVisualizer_.visualize(state);
    heightsReconstructedVisualizer_.visualize(state, sampled_, hidden_);

    std::cout << sampled_.cols() << std::endl;
    std::cout << sampled_.rows() << std::endl;
    std::cout << std::endl;
}

void BobController::changeController(const std::string &controllerType, scalar_t currentTime) {
    gridmap_->waitTillInitialized();
}

bool BobController::isSupported(const std::string &controllerType) {
    if (controllerType == "BOB" || controllerType == "RL") {
        return true;
    }
    return false;
}

tbai_msgs::JointCommandArray BobController::getCommandMessage(scalar_t currentTime, scalar_t dt) {
    auto state = getBobnetState();

    // Do not keep track of gradients
    torch::NoGradGuard no_grad;

    auto ts1 = std::chrono::high_resolution_clock::now();
    at::Tensor nnInput = getNNInput(state, currentTime, dt);
    auto t2 = std::chrono::high_resolution_clock::now();

    cpg_->step(dt);

    ROS_INFO_STREAM_THROTTLE(
        1.0, "NN input computation took: "
                 << std::chrono::duration_cast<std::chrono::microseconds>(t2 - ts1).count() / 1000.0 << " ms");

    std::cout << nnInput.sizes() << std::endl;
    // perform forward pass
    auto ts3 = std::chrono::high_resolution_clock::now();
    at::Tensor out = model_.forward({nnInput.view({1, getNNInputSize()})}).toTensor().squeeze();
    auto t4 = std::chrono::high_resolution_clock::now();

    ROS_INFO_STREAM_THROTTLE(
        1.0,
        "NN forward pass took: " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - ts3).count() / 1000.0
                                 << " ms");

    // action from NN
    at::Tensor action = out.index({Slice(0, COMMAND_SIZE)});

    // reconstructed hidden information
    hidden_ = out.index({Slice(COMMAND_SIZE, None)});

    // unpack action
    at::Tensor phaseOffsets = action.index({Slice(0, 4)});
    vector_t phaseOffsetsVec(4);
    for (size_t i = 0; i < 4; ++i) {
        phaseOffsetsVec[i] = (phaseOffsets[i].item<float>() * ACTION_SCALE);
    }

    at::Tensor jointResiduals = action.index({Slice(0, COMMAND_SIZE)});
    vector_t jointResidualsVec(12);
    for (size_t i = 0; i < 12; ++i) {
        jointResidualsVec[i] = (jointResiduals[i].item<float>() * ACTION_SCALE);
    }

    // compute ik
    auto legHeights = cpg_->legHeights();
    jointAngles2_ = ik_->solve(legHeights) + jointResidualsVec;

    // Send command
    auto ret = getCommandMessage(jointAngles2_);

    // Compute IK again for buffer, TODO: Remove this calculation
    cpg_->step(-dt);
    legHeights = cpg_->legHeights();
    jointAngles2_ = ik_->solve(legHeights);

    // update history buffer
    updateHistory(nnInput, action, state);

    // update central pattern generator
    cpg_->step(dt);

    return ret;
}

tbai_msgs::JointCommandArray BobController::getCommandMessage(const vector_t &jointAngles) {
    tbai_msgs::JointCommandArray commandArray;
    commandArray.joint_commands.resize(jointAngles.size());
    for (size_t i = 0; i < jointAngles.size(); ++i) {
        tbai_msgs::JointCommand &command = commandArray.joint_commands[i];
        command.joint_name = jointNames_[i];
        command.desired_position = jointAngles[i];
        command.desired_velocity = 0.0;
        command.kp = kp_;
        command.kd = kd_;
        command.torque_ff = 0.0;
    }
    return commandArray;
}

void BobController::resetHistory() {
    auto reset = [](auto &history, size_t history_size, size_t item_size) {
        history.clear();
        for (size_t i = 0; i < history_size; ++i) {
            history.push_back(torch::zeros({item_size}));
        }
    };
    reset(historyResiduals_, POSITION_HISTORY_SIZE, POSITION_SIZE);
    reset(historyVelocities_, VELOCITY_HISTORY_SIZE, VELOCITY_SIZE);
    reset(historyActions_, COMMAND_HISTORY_SIZE, COMMAND_SIZE);
}

State BobController::getBobnetState() {
    const vector_t &stateSubscriberState = stateSubscriberPtr_->getLatestRbdState();
    State ret;

    // Base position
    ret.basePositionWorld = stateSubscriberState.segment<3>(0);

    // Base orientation
    Eigen::Quaternion q(stateSubscriberState[6], stateSubscriberState[3], stateSubscriberState[4],
                        stateSubscriberState[5]);
    auto Rwb = q.toRotationMatrix();
    auto Rbw = Rwb.transpose();
    ret.baseOrientationWorld = (State::Vector4() << q.x(), q.y(), q.z(), q.w()).finished();

    // Base linear velocity
    ret.baseLinearVelocityBase = Rbw * stateSubscriberState.segment<3>(19);

    // Base angular velocity
    ret.baseAngularVelocityBase = Rbw * stateSubscriberState.segment<3>(22);

    // Normalized gravity vector
    ret.normalizedGravityBase = Rbw * (vector3_t() << 0, 0, -1).finished();

    // Joint positions
    ret.jointPositions = stateSubscriberState.segment<12>(7);

    // Joint velocities
    ret.jointVelocities = stateSubscriberState.segment<12>(3 + 4 + 12 + 3 + 3);

    // pinocchio state vector
    vector_t pinocchioStateVector = stateSubscriberState.head<19>();

    // Update kinematics
    pinocchio::forwardKinematics(pinocchioModel_, pinocchioData_, pinocchioStateVector);
    pinocchio::updateFramePlacements(pinocchioModel_, pinocchioData_);

    ret.lfFootPositionWorld = pinocchioData_.oMf[pinocchioModel_.getBodyId("LF_FOOT")].translation();
    ret.lhFootPositionWorld = pinocchioData_.oMf[pinocchioModel_.getBodyId("LH_FOOT")].translation();
    ret.rfFootPositionWorld = pinocchioData_.oMf[pinocchioModel_.getBodyId("RF_FOOT")].translation();
    ret.rhFootPositionWorld = pinocchioData_.oMf[pinocchioModel_.getBodyId("RH_FOOT")].translation();

    return ret;
}

at::Tensor BobController::getNNInput(const State &state, scalar_t currentTime, scalar_t dt) {
    at::Tensor input = at::empty(getNNInputSize());

    // Fill individual sections of the nn input tensor
    fillCommand(input, currentTime, dt);
    fillGravity(input, state);
    fillBaseLinearVelocity(input, state);
    fillBaseAngularVelocity(input, state);
    fillJointResiduals(input, state);
    fillJointVelocities(input, state);
    fillHistory(input);
    fillCpg(input);
    fillHeights(input, state);
    return input;
}

/***********************************************************************************************************************/
/***********************************************************************************************************************/
/***********************************************************************************************************************/
void BobController::fillCommand(at::Tensor &input, scalar_t currentTime, scalar_t dt) {
    auto command = refVelGen_->getReferenceVelocity(currentTime, dt);
    input[0] = command.velocity_x;
    input[1] = command.velocity_y;
    input[2] = command.yaw_rate;
}

/***********************************************************************************************************************/
/***********************************************************************************************************************/
/***********************************************************************************************************************/
void BobController::fillGravity(at::Tensor &input, const State &state) {
    input[3] = state.normalizedGravityBase[0] * GRAVITY_SCALE;
    input[4] = state.normalizedGravityBase[1] * GRAVITY_SCALE;
    input[5] = state.normalizedGravityBase[2] * GRAVITY_SCALE;
}

/***********************************************************************************************************************/
/***********************************************************************************************************************/
/***********************************************************************************************************************/
void BobController::fillBaseLinearVelocity(at::Tensor &input, const State &state) {
    input[6] = state.baseLinearVelocityBase[0] * LIN_VEL_SCALE;
    input[7] = state.baseLinearVelocityBase[1] * LIN_VEL_SCALE;
    input[8] = state.baseLinearVelocityBase[2] * LIN_VEL_SCALE;
}

/***********************************************************************************************************************/
/***********************************************************************************************************************/
/***********************************************************************************************************************/
void BobController::fillBaseAngularVelocity(at::Tensor &input, const State &state) {
    input[9] = state.baseAngularVelocityBase[0] * ANG_VEL_SCALE;
    input[10] = state.baseAngularVelocityBase[1] * ANG_VEL_SCALE;
    input[11] = state.baseAngularVelocityBase[2] * ANG_VEL_SCALE;
}

/***********************************************************************************************************************/
/***********************************************************************************************************************/
/***********************************************************************************************************************/
void BobController::fillJointResiduals(at::Tensor &input, const State &state) {
    // fill joint residuals
    for (size_t i = 0; i < 12; ++i) {
        input[12 + i] = (state.jointPositions[i] - jointAngles2_[i]) * JOINT_POS_SCALE;
    }
}

/***********************************************************************************************************************/
/***********************************************************************************************************************/
/***********************************************************************************************************************/
void BobController::fillJointVelocities(at::Tensor &input, const State &state) {
    // fill joint velocities
    for (size_t i = 0; i < 12; ++i) {
        input[24 + i] = state.jointVelocities[i] * JOINT_VEL_SCALE;
    }
}

/***********************************************************************************************************************/
/***********************************************************************************************************************/
/***********************************************************************************************************************/
void BobController::fillHistory(at::Tensor &input) {
    fillHistoryResiduals(input);
    fillHistoryVelocities(input);
    fillHistoryActions(input);
}

/***********************************************************************************************************************/
/***********************************************************************************************************************/
/***********************************************************************************************************************/
void BobController::fillCpg(at::Tensor &input) {
    const size_t startIdx = 36 + POSITION_HISTORY_SIZE * POSITION_SIZE + VELOCITY_HISTORY_SIZE * VELOCITY_SIZE +
                            COMMAND_HISTORY_SIZE * COMMAND_SIZE;
    auto cpgObservation = cpg_->getObservation();
    for (size_t i = 0; i < 8; ++i) {
        input[startIdx + i] = cpgObservation[i];
    }
}

/***********************************************************************************************************************/
/***********************************************************************************************************************/
/***********************************************************************************************************************/
void BobController::fillHeights(at::Tensor &input, const State &state) {
    const size_t startIdx = 36 + POSITION_HISTORY_SIZE * POSITION_SIZE + VELOCITY_HISTORY_SIZE * VELOCITY_SIZE +
                            COMMAND_HISTORY_SIZE * COMMAND_SIZE + 8;

    // Find yaw angle
    quaternion_t q(state.baseOrientationWorld[3], state.baseOrientationWorld[0], state.baseOrientationWorld[1],
                   state.baseOrientationWorld[2]);

    scalar_t yaw = pinocchio::rpy::matrixToRpy(q.toRotationMatrix())[2];

    // Rotate sampling points
    matrix3_t Ryaw = angleaxis_t(yaw, vector3_t::UnitZ()).toRotationMatrix();
    matrix_t rotatedSamplingPoints = Ryaw * gridmap_->samplingPositions_;

    // Replicate sampling points
    sampled_ = rotatedSamplingPoints.replicate<1, 4>();

    auto addOffset = [this](scalar_t x_offset, scalar_t y_offset, size_t idx) mutable {
        size_t blockStart = idx * 52;
        sampled_.block<1, 52>(0, blockStart).array() += x_offset;
        sampled_.block<1, 52>(1, blockStart).array() += y_offset;
    };
    addOffset(state.lfFootPositionWorld[0], state.lfFootPositionWorld[1], 0);
    addOffset(state.lhFootPositionWorld[0], state.lhFootPositionWorld[1], 1);
    addOffset(state.rfFootPositionWorld[0], state.rfFootPositionWorld[1], 2);
    addOffset(state.rhFootPositionWorld[0], state.rhFootPositionWorld[1], 3);

    // sample heights
    gridmap_->atPositions(sampled_);

    // clamp third row between -1 and 1
    sampled_.row(2) = (state.basePositionWorld[2] - sampled_.row(2).array()).array() - 0.5;
    sampled_.row(2) = sampled_.row(2).cwiseMax(-1.0).cwiseMin(1.0) * HEIGHT_MEASUREMENTS_SCALE;

    // Eigen -> torch: https://discuss.pytorch.org/t/data-transfer-between-libtorch-c-and-eigen/54156/6
    float *torchPtr = input.index({Slice(startIdx, None)}).data_ptr<float>();
    Eigen::Map<Eigen::VectorXf> ef(torchPtr, 4 * 52, 1);
    ef = sampled_.row(2).cast<float>();
}

/***********************************************************************************************************************/
/***********************************************************************************************************************/
/***********************************************************************************************************************/
void BobController::fillHistoryResiduals(at::Tensor &input) {
    int ip = mod(historyResidualsIndex_ - 1, POSITION_HISTORY_SIZE);
    const size_t startIdx = 36;
    for (int i = 0; i < POSITION_HISTORY_SIZE; ++i) {
        int idx = mod(ip + i, POSITION_HISTORY_SIZE);
        input.index({Slice(startIdx + i * POSITION_SIZE, startIdx + (i + 1) * POSITION_SIZE)}) = historyResiduals_[idx];
    }
}

/***********************************************************************************************************************/
/***********************************************************************************************************************/
/***********************************************************************************************************************/
void BobController::fillHistoryVelocities(at::Tensor &input) {
    int ip = mod(historyVelocitiesIndex_ - 1, VELOCITY_HISTORY_SIZE);
    const size_t startIdx = 36 + POSITION_HISTORY_SIZE * POSITION_SIZE;
    for (int i = 0; i < VELOCITY_HISTORY_SIZE; ++i) {
        int idx = mod(ip + i, VELOCITY_HISTORY_SIZE);
        input.index({Slice(startIdx + i * VELOCITY_SIZE, startIdx + (i + 1) * VELOCITY_SIZE)}) =
            historyVelocities_[idx];
    }
}

/***********************************************************************************************************************/
/***********************************************************************************************************************/
/***********************************************************************************************************************/
void BobController::fillHistoryActions(at::Tensor &input) {
    int ip = mod(historyActionsIndex_ - 1, COMMAND_HISTORY_SIZE);
    const size_t startIdx = 36 + POSITION_HISTORY_SIZE * POSITION_SIZE + VELOCITY_HISTORY_SIZE * VELOCITY_SIZE;
    for (int i = 0; i < COMMAND_HISTORY_SIZE; ++i) {
        int idx = mod(ip + i, COMMAND_HISTORY_SIZE);
        input.index({Slice(startIdx + i * COMMAND_SIZE, startIdx + (i + 1) * COMMAND_SIZE)}) = historyActions_[idx];
    }
}

// void BobController::generateSamplingPositions() {
//     std::vector<scalar_t> Ns = {6, 8, 10, 12, 16};
//     std::vector<scalar_t> rs = {0.1, 0.3, 0.5, 0.7, 0.9};

//     samplingPositions_ = matrix_t::Zero(3, std::accumulate(Ns.begin(), Ns.end(), 0));

//     size_t idx = 0;
//     for (int i = 0; i < Ns.size(); ++i) {
//         scalar_t r = rs[i];
//         scalar_t N = Ns[i];
//         for (int j = 0; j < N; ++j) {
//             scalar_t theta = 2 * M_PI * j / N;
//             scalar_t x = r * std::cos(theta);
//             scalar_t y = r * std::sin(theta);
//             samplingPositions_(0, idx) = x;
//             samplingPositions_(1, idx) = y;
//             ++idx;
//         }
//     }
// }

void BobController::updateHistory(const at::Tensor &input, const at::Tensor &action, const State &state) {
    // update position history
    for (int i = 0; i < 12; ++i) {
        historyResiduals_[historyResidualsIndex_][i] = (state.jointPositions[i] - jointAngles2_[i]) * JOINT_POS_SCALE;
    }
    historyResidualsIndex_ = (historyResidualsIndex_ + 1) % POSITION_HISTORY_SIZE;

    // update velocity history
    historyVelocities_[historyVelocitiesIndex_] = input.index({jointVelocitiesSlice_});
    historyVelocitiesIndex_ = (historyVelocitiesIndex_ + 1) % VELOCITY_HISTORY_SIZE;

    // update action history
    historyActions_[historyActionsIndex_] = action;
    historyActionsIndex_ = (historyActionsIndex_ + 1) % COMMAND_HISTORY_SIZE;
}

StateVisualizer::StateVisualizer() {
    // Load odom frame name
    odomFrame_ = tbai::core::fromRosConfig<std::string>("odom_frame");

    // Load base frame name
    baseFrame_ = tbai::core::fromRosConfig<std::string>("base_name");

    // Load joint names
    jointNames_ = tbai::core::fromRosConfig<std::vector<std::string>>("joint_names");

    // Setup state publisher
    std::string urdfString;
    if (!ros::param::get("/robot_description", urdfString)) {
        throw std::runtime_error("Failed to get param /robot_description");
    }

    KDL::Tree kdlTree;
    kdl_parser::treeFromString(urdfString, kdlTree);
    robotStatePublisherPtr_.reset(new robot_state_publisher::RobotStatePublisher(kdlTree));
    robotStatePublisherPtr_->publishFixedTransforms(true);

    ROS_INFO("StateVisualizer initialized");
}

void StateVisualizer::visualize(const State &state) {
    ros::Time timeStamp = ros::Time::now();
    publishOdomTransform(timeStamp, state);
    publishJointAngles(timeStamp, state);
}

void StateVisualizer::publishOdomTransform(const ros::Time &timeStamp, const State &state) {
    geometry_msgs::TransformStamped baseToWorldTransform;
    baseToWorldTransform.header.stamp = timeStamp;
    baseToWorldTransform.header.frame_id = odomFrame_;
    baseToWorldTransform.child_frame_id = baseFrame_;

    baseToWorldTransform.transform.translation.x = state.basePositionWorld[0];
    baseToWorldTransform.transform.translation.y = state.basePositionWorld[1];
    baseToWorldTransform.transform.translation.z = state.basePositionWorld[2];

    baseToWorldTransform.transform.rotation.x = state.baseOrientationWorld[0];
    baseToWorldTransform.transform.rotation.y = state.baseOrientationWorld[1];
    baseToWorldTransform.transform.rotation.z = state.baseOrientationWorld[2];
    baseToWorldTransform.transform.rotation.w = state.baseOrientationWorld[3];

    tfBroadcaster_.sendTransform(baseToWorldTransform);
}

void StateVisualizer::publishJointAngles(const ros::Time &timeStamp, const State &state) {
    std::map<std::string, double> positions;
    for (int i = 0; i < jointNames_.size(); ++i) {
        positions[jointNames_[i]] = state.jointPositions[i];
    }
    robotStatePublisherPtr_->publishTransforms(positions, timeStamp);
}

HeightsReconstructedVisualizer::HeightsReconstructedVisualizer() {
    ros::NodeHandle nh;
    std::string markerTopic = tbai::core::fromRosConfig<std::string>("marker_topic");
    markerPublisher_ = nh.advertise<visualization_msgs::MarkerArray>(markerTopic, 1);

    odomFrame_ = tbai::core::fromRosConfig<std::string>("odom_frame");
}

void HeightsReconstructedVisualizer::visualize(const State &state, const matrix_t &sampled,
                                               const at::Tensor &nnPointsReconstructed) {
    ros::Time timeStamp = ros::Time::now();

    publishMarkers(timeStamp, sampled, {1.0, 0.0, 0.0}, "ground_truth",
                   [&](size_t id) { return -(sampled(2, id) / 1.0 + 0.5 - state.basePositionWorld[2]); });

    publishMarkers(timeStamp, sampled, {0.0, 0.0, 1.0}, "nn_reconstructed", [&](size_t id) {
        float out = -(nnPointsReconstructed[id].item<float>() / 1.0 + 0.5 - state.basePositionWorld[2]);
        return static_cast<scalar_t>(out);
    });
}

void HeightsReconstructedVisualizer::publishMarkers(const ros::Time &timeStamp, const matrix_t &sampled,
                                                    const std::array<float, 3> &rgb,
                                                    const std::string &markerNamePrefix,
                                                    std::function<scalar_t(size_t)> heightFunction) {
    const size_t nPoints = sampled.cols();
    const size_t pointsPerLeg = nPoints / 4;

    constexpr uint32_t shape = visualization_msgs::Marker::SPHERE;
    visualization_msgs::MarkerArray markerArray;
    markerArray.markers.reserve(nPoints);

    for (size_t leg_idx = 0; leg_idx < 4; ++leg_idx) {
        const std::string ns = markerNamePrefix + std::to_string(leg_idx);
        for (size_t i = 0; i < pointsPerLeg; ++i) {
            size_t id = leg_idx * pointsPerLeg + i;
            visualization_msgs::Marker marker;
            marker.header.frame_id = odomFrame_;
            marker.header.stamp = timeStamp;
            marker.ns = ns;
            marker.id = id;
            marker.type = shape;
            marker.action = visualization_msgs::Marker::ADD;

            marker.pose.position.x = sampled(0, id);
            marker.pose.position.y = sampled(1, id);
            marker.pose.position.z = heightFunction(id);

            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;

            marker.scale.x = 0.03;
            marker.scale.y = 0.03;
            marker.scale.z = 0.03;

            marker.color.r = rgb[0];
            marker.color.g = rgb[1];
            marker.color.b = rgb[2];
            marker.color.a = 1.0;

            markerArray.markers.push_back(marker);
        }
    }

    markerPublisher_.publish(markerArray);
}

}  // namespace rl
}  // namespace tbai
