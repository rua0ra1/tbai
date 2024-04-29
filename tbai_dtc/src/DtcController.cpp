// clang-format off
#include <pinocchio/fwd.hpp>
// clang-format on

#include <tbai_core/Rotations.hpp>
#include "tbai_dtc/DtcController.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <ocs2_legged_robot/gait/LegLogic.h>
#include <ocs2_core/misc/LinearInterpolation.h>

#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <ocs2_robotic_tools/common/RotationDerivativesTransforms.h>
#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_centroidal_model/PinocchioCentroidalDynamics.h>

#include <ocs2_msgs/mpc_target_trajectories.h>
#include "ocs2_ros_interfaces/common/RosMsgConversions.h"
#include <ocs2_legged_robot/gait/MotionPhaseDefinition.h>

#include <ocs2_core/reference/TargetTrajectories.h>

#include <pinocchio/algorithm/centroidal.hpp>
#include <string>
#include <tbai_core/Asserts.hpp>
#include <vector>

#include <tbai_core/Utils.hpp>

namespace tbai {

namespace dtc {

DtcController::DtcController(const std::shared_ptr<tbai::core::StateSubscriber> &stateSubscriber)
    : stateSubscriberPtr_(stateSubscriber), mrt_("legged_robot") {
    initTime_ = tbai::core::getEpochStart();

    ros::NodeHandle nh;
    mrt_.launchNodes(nh);
    setupPinocchioModel();

    std::string taskFile;
    ros::param::get("taskFile", taskFile);

    std::string urdfFile;
    ros::param::get("urdfFile", urdfFile);

    std::string referenceFile;
    ros::param::get("referenceFile", referenceFile);

    ROS_INFO_STREAM("[DtcController] Setting up legged interface");

    leggedInterface_ = std::make_unique<ocs2::legged_robot::LeggedRobotInterface>(taskFile, urdfFile, referenceFile);
    ROS_INFO_STREAM("[DtcController] Setting up pinocchio interface");
    pinocchioInterface_ = std::make_unique<ocs2::PinocchioInterface>(leggedInterface_->getPinocchioInterface());
    ROS_INFO_STREAM("[DtcController] Setting up centroidal model mapping");
    centroidalModelMapping_ =
        std::make_unique<ocs2::CentroidalModelPinocchioMapping>(leggedInterface_->getCentroidalModelInfo());

    ROS_INFO_STREAM("[DtcController] Setting up centroidal model mapping");
    centroidalModelMapping_->setPinocchioInterface(*pinocchioInterface_);

    ROS_INFO_STREAM("[DtcController] Setting up end effector kinematics");
    std::vector<std::string> ee = {"LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"};
    endEffectorKinematics_ =
        std::make_unique<ocs2::PinocchioEndEffectorKinematics>(*pinocchioInterface_, *centroidalModelMapping_, ee);

    ROS_INFO_STREAM("[DtcController] Setting up end effector kinematics");
    endEffectorKinematics_->setPinocchioInterface(*pinocchioInterface_);

    ROS_INFO_STREAM("[DtcController] Setting up default Dof positions");

    defaultDofPositions_ = (vector_t(12) << 0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8).finished();
    ROS_INFO_STREAM("[DtcController] Setup done");

    refVelGen_ = tbai::reference::getReferenceVelocityGeneratorUnique(nh);
    refPub_ = nh.advertise<ocs2_msgs::mpc_target_trajectories>("legged_robot_mpc_target", 1, false);

    ocs2::CentroidalModelPinocchioMapping pinocchioMapping(leggedInterface_->getCentroidalModelInfo());
    ocs2::PinocchioEndEffectorKinematics endEffectorKinematics(leggedInterface_->getPinocchioInterface(),
                                                               pinocchioMapping,
                                                               leggedInterface_->modelSettings().contactNames3DoF);
    // auto leggedRobotVisualizer = std::make_shared<LeggedRobotVisualizer>(
    //    interface.getPinocchioInterface(), interface.getCentroidalModelInfo(), endEffectorKinematics, nodeHandle);
    visualizer_ = std::make_unique<ocs2::legged_robot::LeggedRobotVisualizer>(
        leggedInterface_->getPinocchioInterface(), leggedInterface_->getCentroidalModelInfo(), endEffectorKinematics,
        nh);

    initializeObservations();

    // Load dtc model
    std::string dtcModelFile = "/home/kuba/fun/tbai/src/tbai/tbai_dtc/models/model_deploy_jitted.pt";
    try {
        ROS_INFO_STREAM("[DtcController] Loading torch model from: " << dtcModelFile);
        dtcModel_ = torch::jit::load(dtcModelFile);
    } catch (const c10::Error &e) {
        std::cerr << "Could not load model from: " << dtcModelFile << std::endl;
        throw std::runtime_error("Could not load model");
    }

    action_ = torch::zeros({12});
    mpcRate_ = 20.0;
}

void DtcController::publishReference(scalar_t currentTime, scalar_t dt) {
    dt = 0.1;
    auto refVel = refVelGen_->getReferenceVelocity(currentTime, 0.1);
    auto x_vel = refVel.velocity_x;
    auto y_vel = refVel.velocity_y;
    auto yaw_rate = refVel.yaw_rate;
    constexpr scalar_t horizon = 1.0;
    const scalar_t finalTime = currentTime + horizon;

    std::cout << "Publishing reference..."
              << "vx: " << x_vel << " vy: " << y_vel << " yaw_rate: " << yaw_rate << std::endl;

    if (dt == 0) {
        dt = 0.1;
    }

    auto currentObservation = generateSystemObservation();

    // auto initState = initialState_;
    auto currentState = currentObservation.state;
    auto initState = currentState;

    const scalar_t v_x = x_vel;
    const scalar_t v_y = y_vel;
    const scalar_t w_z = yaw_rate;

    ocs2::scalar_array_t timeTrajectory;
    ocs2::vector_array_t stateTrajectory;
    ocs2::vector_array_t inputTrajectory;

    // Insert initial time, state and input
    timeTrajectory.push_back(currentTime);
    stateTrajectory.push_back(initState);
    inputTrajectory.push_back(vector_t().setZero(24));

    // Complete the rest of the trajectory
    scalar_t time = currentTime;
    vector_t nextState = initState;
    nextState.tail<12>() = defaultDofPositions_;
    nextState.segment<6>(0) = currentState.segment<6>(0);
    nextState.segment<6>(6) = currentState.segment<6>(6);
    nextState(8) = 0.54;  // z position
    nextState(10) = 0.0;  // roll
    nextState(11) = 0.0;  // pitch
    nextState(2) = 0.0;   // z velocity
    while (time < finalTime) {
        time += dt;

        const scalar_t yaw = nextState(9);
        const scalar_t cy = std::cos(yaw);
        const scalar_t sy = std::sin(yaw);

        const scalar_t dx = (cy * v_x - sy * v_y) * dt;
        const scalar_t dy = (sy * v_x + cy * v_y) * dt;
        const scalar_t dw = w_z * dt;

        nextState(0) = dx / dt;
        nextState(1) = dy / dt;

        nextState(6) += dx;
        nextState(7) += dy;
        nextState(9) += dw;

        timeTrajectory.push_back(time);
        stateTrajectory.push_back(nextState);
        inputTrajectory.push_back(vector_t().setZero(24));
    }

    ocs2::TargetTrajectories t(timeTrajectory, stateTrajectory, inputTrajectory);
    const auto mpcTargetTrajectoriesMsg = ocs2::ros_msg_conversions::createTargetTrajectoriesMsg(t);
    refPub_.publish(mpcTargetTrajectoriesMsg);
}

tbai_msgs::JointCommandArray DtcController::getCommandMessage(scalar_t currentTime, scalar_t dt) {
    std::vector<std::string> jointNames = {"LF_HAA", "LF_HFE", "LF_KFE", "LH_HAA", "LH_HFE", "LH_KFE",
                                           "RF_HAA", "RF_HFE", "RF_KFE", "RH_HAA", "RH_HFE", "RH_KFE"};

    std::vector<scalar_t> jointAngles = {0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8};

    currentTime = ros::Time::now().toSec() - initTime_;

    tbai_msgs::JointCommandArray jointCommandArray;
    jointCommandArray.joint_commands.resize(12);
    for (int i = 0; i < 12; ++i) {
        auto &command = jointCommandArray.joint_commands[i];
        command.joint_name = jointNames[i];
        command.desired_position = jointAngles[i];
        command.desired_velocity = 0.0;
        command.torque_ff = 0.0;
        command.kp = 400;
        command.kd = 10;
    }

    std::cout << "DTC controller, Current time:" << currentTime << std::endl;

    mrt_.spinMRT();
    mrt_.updatePolicy();

    ocs2::PrimalSolution policy = mrt_.getPolicy();
    computeObservation(policy, currentTime);

    timeSinceLastMpcUpdate_ += dt;
    if (timeSinceLastMpcUpdate_ >= 1.0 / mpcRate_) {
        // resetMpc();
        publishReference(currentTime, dt);
        setObservation();
        timeSinceLastMpcUpdate_ = 0.0;
    }

    // Compute input for the model
    torch::Tensor input = torch::zeros({MODEL_INPUT_SIZE});
    fillObsTensor(input);

    // Perform forward pass
    auto t_start = std::chrono::high_resolution_clock::now();
    // std::cout << "========================== START ===========================" << std::endl;
    // std::cout << input << std::endl;
    // std::cout << "========================== STOP ===========================" << std::endl;
    auto output = dtcModel_.forward({input.view({1, MODEL_INPUT_SIZE})}).toTensor().squeeze();
    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1e3;

    // ROS_INFO_STREAM("[DtcController] Forward pass duration: " << duration_ms << " ms");
    action_ = output;

    // Unpack actions
    vector_t action = vector_t().setZero(12);
    for (int i = 0; i < 12; ++i) {
        action(i) = output[i].item<float>() * ACTION_SCALE + kk_(i);
        // action(i) = kk_(i);
    }

    for (int i = 0; i < 12; ++i) {
        scalar_t angle = action(i);
        auto &jointName = jointNames[i];
        // ROS_INFO_STREAM("[DtcController] Joint " << jointName << " angle: " << angle);
    }

    for (int i = 0; i < 12; ++i) {
        auto &command = jointCommandArray.joint_commands[i];
        command.joint_name = jointNames[i];
        command.desired_position = action(i);
        command.desired_velocity = 0.0;
        command.torque_ff = 0.0;
        command.kp = 80;
        command.kd = 2;
    }

    return jointCommandArray;
}

void DtcController::fillObsTensor(torch::Tensor &tensor) {
    // Linear velocity
    tensor[0] = obsLinearVelocity_(0) * LIN_VEL_SCALE;
    tensor[1] = obsLinearVelocity_(1) * LIN_VEL_SCALE;
    tensor[2] = obsLinearVelocity_(2) * LIN_VEL_SCALE;

    // Angular velocity
    tensor[3] = obsAngularVelocity_(0) * ANG_VEL_SCALE;
    tensor[4] = obsAngularVelocity_(1) * ANG_VEL_SCALE;
    tensor[5] = obsAngularVelocity_(2) * ANG_VEL_SCALE;

    // Gravity
    tensor[6] = obsProjectedGravity_(0);
    tensor[7] = obsProjectedGravity_(1);
    tensor[8] = obsProjectedGravity_(2);

    // Command
    tensor[9] = obsCommand_(0);
    tensor[10] = obsCommand_(1);
    tensor[11] = obsCommand_(2);

    // Joint positions
    size_t startIdx = 12;
    for (int i = 0; i < 12; ++i) {
        tensor[startIdx + i] = obsJointPositions_(i) * DOF_POS_SCALE;
    }
    startIdx += 12;

    // Joint velocities
    for (int i = 0; i < 12; ++i) {
        tensor[startIdx + i] = obsJointVelocities_(i) * DOF_VEL_SCALE;
    }
    startIdx += 12;

    // Last action
    for (int i = 0; i < 12; ++i) {
        tensor[startIdx + i] = obsPastAction_(i);
    }
    startIdx += 12;

    // Planar footholds
    for (int i = 0; i < 8; ++i) {
        tensor[startIdx + i] = obsDesiredFootholds_(i);
    }
    startIdx += 8;

    // Desired joint angles
    for (int i = 0; i < 12; ++i) {
        tensor[startIdx + i] = obsDesiredJointAngles_(i);
    }
    startIdx += 12;

    // Current desired joint angles
    for (int i = 0; i < 12; ++i) {
        tensor[startIdx + i] = obsCurrentDesiredJointAngles_(i);
    }
    startIdx += 12;

    // Desired contacts
    for (int i = 0; i < 4; ++i) {
        tensor[startIdx + i] = obsDesiredContacts_(i);
    }
    std::cout << obsDesiredContacts_.transpose() << std::endl;
    startIdx += 4;

    // Time left in phase
    for (int i = 0; i < 4; ++i) {
        tensor[startIdx + i] = obsTimeLeftInPhase_(i);
    }
    startIdx += 4;

    // Desired base pos
    for (int i = 0; i < 3; ++i) {
        tensor[startIdx + i] = obsDesiredBasePosition_(i);
    }
    startIdx += 3;

    // Desired base orientation
    for (int i = 0; i < 4; ++i) {
        tensor[startIdx + i] = obsDesiredBaseOrientation_(i);
    }
    startIdx += 4;

    // Desired base linear velocity
    for (int i = 0; i < 3; ++i) {
        tensor[startIdx + i] = obsDesiredBaseLinearVelocity_(i);
    }
    startIdx += 3;

    // Desired base angular velocity
    for (int i = 0; i < 3; ++i) {
        tensor[startIdx + i] = obsDesiredBaseAngularVelocity_(i);
    }
    startIdx += 3;

    // Desired base linear acceleration
    for (int i = 0; i < 3; ++i) {
        tensor[startIdx + i] = obsDesiredBaseLinearAcceleration_(i);
    }
    startIdx += 3;

    // Desired base angular acceleration
    for (int i = 0; i < 3; ++i) {
        tensor[startIdx + i] = obsDesiredBaseAngularAcceleration_(i);
    }
    startIdx += 3;

    TBAI_ASSERT(startIdx == MODEL_INPUT_SIZE, "Invalid tensor size");
}

void DtcController::visualize() {
    std::cout << "Visualizing..." << std::endl;
    visualizer_->update(generateSystemObservation(), mrt_.getPolicy(), mrt_.getCommand());
}

void DtcController::changeController(const std::string &controllerType, scalar_t currentTime) {
    resetMpc();
}

bool DtcController::isSupported(const std::string &controllerType) {
    return controllerType == "DTC";
}

ocs2::SystemObservation DtcController::generateSystemObservation() const {
    const tbai::vector_t &rbdState = stateSubscriberPtr_->getLatestRbdState();

    // Set observation time
    ocs2::SystemObservation observation;
    observation.time = stateSubscriberPtr_->getLatestRbdStamp().toSec() - initTime_;

    // Set mode
    observation.mode = 14;  // This is not important

    vector3_t rpyocs2 = (vector3_t() << rbdState(0), rbdState(1), rbdState(2)).finished();
    // matrix3_t R_world_base = tbai::core::rpy2mat(rpy);
    matrix3_t R_world_base = tbai::core::ocs2rpy2quat(rpyocs2).toRotationMatrix();
    vector3_t rpy = tbai::core::mat2rpy(R_world_base);
    matrix3_t R_base_world = R_world_base.transpose();

    vector_t state = vector_t().setZero(24);

    // state.segment<3>(0) = R_world_base * rbdState.segment<3>(9);  // v_com in world frame
    // state.segment<3>(3) = vector_t().setZero(3);                  // normalized angular momentum
    state.segment<3>(6) = rbdState.segment<3>(3);      // com position
    state.segment<3>(9) = rpy.reverse();               // ypr
    state.segment<12>(12) = rbdState.segment<12>(12);  // joint angles

    TBAI_ASSERT(state.segment<3>(3).norm() < 1e5, "COM position is too large");

    vector_t input = vector_t().setZero(24);
    // input.segment<12>(0) = vector_t().setZero(12);     // ground reaction forces
    input.segment<12>(12) = rbdState.segment<12>(24);  // joint velocities

    TBAI_ASSERT(input.segment<12>(0).norm() < 1e5, "GRF is too large");

    observation.state = std::move(state);
    observation.input = std::move(input);

    return observation;
}

void DtcController::setupPinocchioModel() {
    // Load urdf model path
    std::string urdfFile;
    ros::param::get("urdfFile", urdfFile);

    // Prepare world -> base joint model
    pinocchio::JointModelComposite jointComposite(2);
    jointComposite.addJoint(pinocchio::JointModelTranslation());
    jointComposite.addJoint(pinocchio::JointModelSphericalZYX());

    // Load model
    pinocchio::urdf::buildModel(urdfFile, jointComposite, model_);

    // Prepare data
    data_ = pinocchio::Data(model_);
}

void DtcController::resetMpc() {
    // Generate initial observation
    stateSubscriberPtr_->waitTillInitialized();
    auto initialObservation = generateSystemObservation();
    // initialObservation.state.segment<3>(0) = vector_t::Zero(3);
    initialObservation.state.segment<3>(3) = vector_t::Zero(3);
    initialObservation.state(6 + 2) = 0.54;
    initialObservation.state.segment<12>(12) = defaultDofPositions_;
    initialObservation.input.segment<24>(0) = vector_t::Zero(24);

    const ocs2::TargetTrajectories initTargetTrajectories({0.0}, {initialObservation.state},
                                                          {initialObservation.input});
    mrt_.resetMpcNode(initTargetTrajectories);

    initialObservation = generateSystemObservation();

    while (!mrt_.initialPolicyReceived() && ros::ok()) {
        ROS_INFO("Waiting for initial policy...");
        ros::spinOnce();
        mrt_.spinMRT();
        initialObservation = generateSystemObservation();
        mrt_.setCurrentObservation(initialObservation);
        ros::Duration(0.1).sleep();
    }

    ROS_INFO("Initial policy received.");
}

void DtcController::setObservation() {
    mrt_.setCurrentObservation(generateSystemObservation());
}

void DtcController::computeObservation(const ocs2::PrimalSolution &policy, scalar_t time) {
    auto &state = stateSubscriberPtr_->getLatestRbdState();

    obsLinearVelocity_ = state.segment<3>(9);   // COM velocity - already expessed in base frame
    obsAngularVelocity_ = state.segment<3>(6);  // Angular velocity - already expessed in base frame

    // Compute projected gravity
    auto rpy = state.segment<3>(0);
    auto R_world_base = tbai::core::ocs2rpy2quat(rpy).toRotationMatrix();
    auto R_base_world = R_world_base.transpose();
    obsProjectedGravity_ = R_base_world * vector3_t(0, 0, -1.0);

    // Current joint positions
    obsJointPositions_ = state.segment<12>(12) - defaultDofPositions_;

    // Current joint velocities
    obsJointVelocities_ = state.segment<12>(24);

    // Update obsPastAction_ during model evaluation
    obsPastAction_ = vector_t().setZero(12);
    for (int i = 0; i < 12; ++i) {
        obsPastAction_(i) = action_[i].item<float>();
    }

    computeCommandObservation(time);

    // Compute desired contacts
    computeDesiredContacts(time);

    // Compute time left in phases
    computeTimeLeftInPhases(time);

    // Compute desired footholds
    computeDesiredFootholds(time);

    computeCurrentDesiredJointAngles(time + 1 / 50);  // TODO: time + some time delta
    computeBaseObservation(time + 1 / 50);            // TODO: time + some time delta ?????
}

void DtcController::computeCommandObservation(scalar_t time) {
    auto refvel = refVelGen_->getReferenceVelocity(time, 0.1);
    obsCommand_(0) = refvel.velocity_x;
    obsCommand_(1) = refvel.velocity_y;
    obsCommand_(2) = refvel.yaw_rate;

    std::cout << "Command: " << obsCommand_.transpose() << std::endl;
}

void DtcController::computeDesiredContacts(scalar_t time) {
    auto &solution = mrt_.getPolicy();
    auto &modeSchedule = solution.modeSchedule_;
    size_t mode = modeSchedule.modeAtTime(time);
    auto contactFlags = ocs2::legged_robot::modeNumber2StanceLeg(mode);
    std::swap(contactFlags[1], contactFlags[2]);
    TBAI_ASSERT(contactFlags.size() == 4, "Contact flags must have size 4");
    for (int i = 0; i < 4; ++i) {
        obsDesiredContacts_(i) = contactFlags[i];  // Assume LF, LH, RF, RH ordering
    }
}

void DtcController::computeTimeLeftInPhases(scalar_t time) {
    auto &solution = mrt_.getPolicy();
    auto &modeSchedule = solution.modeSchedule_;

    auto currentContacts = obsDesiredContacts_;
    auto contactPhases = ocs2::legged_robot::getContactPhasePerLeg(time, modeSchedule);
    auto swingPhases = ocs2::legged_robot::getSwingPhasePerLeg(time, modeSchedule);

    TBAI_ASSERT(contactPhases.size() == 4, "Contact phases must have size 4");
    TBAI_ASSERT(swingPhases.size() == 4, "Swing phases must have size 4");

    for (int i = 0; i < 4; ++i) {
        if (currentContacts(i)) {
            auto legPhase = contactPhases[i];
            obsTimeLeftInPhase_(i) = (1.0 - legPhase.phase) * legPhase.duration;
        } else {
            auto legPhase = swingPhases[i];
            obsTimeLeftInPhase_(i) = (1.0 - legPhase.phase) * legPhase.duration;
        }

        if (std::isnan(obsTimeLeftInPhase_(i))) {
            obsTimeLeftInPhase_(i) = 0.0;
        }
    }
}

void DtcController::computeDesiredFootholds(scalar_t time) {
    auto &solution = mrt_.getPolicy();
    auto &modeSchedule = solution.modeSchedule_;

    // current state
    auto currentRbdState = stateSubscriberPtr_->getLatestRbdState();
    auto rpy = tbai::core::mat2rpy(tbai::core::ocs2rpy2quat(currentRbdState.segment<3>(0)).toRotationMatrix());
    auto currentJointAngles = currentRbdState.segment<12>(12);
    // set roll and pitch to zero

    rpy[0] = 0.0;
    rpy[1] = 0.0;
    auto R_world_base = tbai::core::rpy2mat(rpy);
    auto R_base_world = R_world_base.transpose();
    vector3_t basePosition = currentRbdState.segment<3>(3);

    obsDesiredJointAngles_ = -currentJointAngles;

    for (int i = 0; i < 4; ++i) {
        float timeLeft = obsTimeLeftInPhase_(i);
        auto state = ocs2::LinearInterpolation::interpolate(time + timeLeft, solution.timeTrajectory_,
                                                            solution.stateTrajectory_);
        auto jointAngles = state.segment<12>(12);

        // Update desired joint angles
        obsDesiredJointAngles_.segment<3>(3 * i) += jointAngles.segment<3>(3 * i);

        // Compute forward kinematics
        auto &pinocchioMapping = *centroidalModelMapping_;
        auto &interface = *pinocchioInterface_;
        auto q = pinocchioMapping.getPinocchioJointPosition(state);
        pinocchio::forwardKinematics(interface.getModel(), interface.getData(), q);
        pinocchio::updateFramePlacements(interface.getModel(), interface.getData());

        // Update end effector kinematics
        auto &endEffector = *endEffectorKinematics_;
        auto positions = endEffector.getPosition(vector_t());

        // Update desired footholds
        vector3_t temp = R_base_world * (positions[i] - basePosition);
        obsDesiredFootholds_.segment<2>(2 * i) = temp.head<2>();  // take only x and y
    }

    std::cout << obsDesiredFootholds_.transpose() << std::endl;
}

void DtcController::computeCurrentDesiredJointAngles(scalar_t time) {
    // Compute desired joint angles
    auto &solution = mrt_.getPolicy();
    // std::cout << "Current time: " << time << std::endl;
    // std::cout << "Solution time: " << solution.timeTrajectory_.front() << std::endl;
    // std::cout << std::endl;
    auto state = ocs2::LinearInterpolation::interpolate(time, solution.timeTrajectory_, solution.stateTrajectory_);
    auto jointAngles = state.segment<12>(12);
    obsCurrentDesiredJointAngles_ = jointAngles - defaultDofPositions_;
    kk_ = defaultDofPositions_;
}

void DtcController::computeBaseObservation(scalar_t time) {
    auto &solution = mrt_.getPolicy();
    auto state = stateSubscriberPtr_->getLatestRbdState();
    vector3_t rpy = state.segment<3>(0);
    matrix3_t R_world_base = tbai::core::ocs2rpy2quat(rpy).toRotationMatrix();
    matrix3_t R_base_world = R_world_base.transpose();
    quaternion_t quatCurrent = tbai::core::ocs2rpy2quat(rpy);

    vector3_t rpyZero = tbai::core::mat2rpy(R_world_base);
    rpy[0] = 0.0;
    rpy[1] = 0.0;
    matrix3_t R_world_base_zero = tbai::core::rpy2mat(rpy);
    matrix3_t R_base_world_zero = R_world_base_zero.transpose();

    // Desired base position
    vector3_t desiredBasePosition =
        ocs2::LinearInterpolation::interpolate(time, solution.timeTrajectory_, solution.stateTrajectory_).segment<3>(6);
    vector3_t currentBasePosition = state.segment<3>(3);
    vector3_t basePositionDiff = desiredBasePosition - currentBasePosition;
    obsDesiredBasePosition_ = R_base_world_zero * basePositionDiff;  // expressed in base frame

    // Desired base orientation
    vector3_t desiredBaseEulerAngles =
        ocs2::LinearInterpolation::interpolate(time, solution.timeTrajectory_, solution.stateTrajectory_).segment<3>(9);
    quaternion_t quatDesired =
        Eigen::AngleAxis<scalar_t>(desiredBaseEulerAngles(0), Eigen::Matrix<scalar_t, 3, 1>::UnitZ()) *
        Eigen::AngleAxis<scalar_t>(desiredBaseEulerAngles(1), Eigen::Matrix<scalar_t, 3, 1>::UnitY()) *
        Eigen::AngleAxis<scalar_t>(desiredBaseEulerAngles(2), Eigen::Matrix<scalar_t, 3, 1>::UnitX());
    quaternion_t check_q = tbai::core::rpy2quat(desiredBaseEulerAngles.reverse());  // reverse because we want rpy order

    if (check_q.x() != quatDesired.x() || check_q.y() != quatDesired.y() || check_q.z() != quatDesired.z() ||
        check_q.w() != quatDesired.w()) {
        std::cout << "Desired quaternion: " << quatDesired.x() << " " << quatDesired.y() << " " << quatDesired.z()
                  << " " << quatDesired.w() << std::endl;
        std::cout << "Check quaternion: " << check_q.x() << " " << check_q.y() << " " << check_q.z() << " "
                  << check_q.w() << std::endl;
    }

    quaternion_t quatDiff =
        quatDesired *
        quatCurrent.conjugate();  // conjugate is enough, the quaternion is normalized <=> conjugate == inverse
    obsDesiredBaseOrientation_ =
        (vector_t(4) << quatDiff.x(), quatDiff.y(), quatDiff.z(), quatDiff.w()).finished();  // x, y, z, w

    // Get desired state and input
    vector_t desiredState =
        ocs2::LinearInterpolation::interpolate(time, solution.timeTrajectory_, solution.stateTrajectory_);
    vector_t desiredInput =
        ocs2::LinearInterpolation::interpolate(time, solution.timeTrajectory_, solution.inputTrajectory_);

    // Update centroidal dynamics
    auto &pinocchioInterface = *pinocchioInterface_;
    auto &pinocchioMapping = *centroidalModelMapping_;
    auto &modelInfo = pinocchioMapping.getCentroidalModelInfo();
    vector_t q = pinocchioMapping.getPinocchioJointPosition(desiredState);
    ocs2::updateCentroidalDynamics(pinocchioInterface, modelInfo, q);

    // Compute desired base linear velocity
    vector3_t desiredBaseLinearVelocity =
        R_base_world * pinocchioMapping.getPinocchioJointVelocity(desiredState, desiredInput).segment<3>(0);
    obsDesiredBaseLinearVelocity_ =
        desiredBaseLinearVelocity;  // desired base linear velocity expressed in the base frame

    // Compute desired base angular velocity
    vector3_t desiredEulerAngleDerivatives = pinocchioMapping.getPinocchioJointVelocity(desiredState, desiredInput)
                                                 .segment<3>(3);  // TODO: convert to local frame
    vector3_t desiredBaseAngularVelocity =
        R_base_world * ocs2::getGlobalAngularVelocityFromEulerAnglesZyxDerivatives<scalar_t>(
                           desiredBaseEulerAngles, desiredEulerAngleDerivatives);
    obsDesiredBaseAngularVelocity_ = desiredBaseAngularVelocity;  // base angular velocity expressed in the base frame

    // Compute desired base linear acceleration
    const scalar_t robotMass = modelInfo.robotMass;
    auto Ag =
        ocs2::getCentroidalMomentumMatrix(pinocchioInterface);  // centroidal momentum matrix as in (h = A * q_dot)
    vector_t h_dot_normalized = ocs2::getNormalizedCentroidalMomentumRate(pinocchioInterface, modelInfo, desiredInput);
    vector_t h_dot = robotMass * h_dot_normalized;

    // pinocchio position and velocity
    vector_t v = pinocchioMapping.getPinocchioJointVelocity(desiredState, desiredInput);
    matrix_t Ag_dot = pinocchio::dccrba(pinocchioInterface.getModel(), pinocchioInterface.getData(), q, v);

    Eigen::Matrix<scalar_t, 6, 6> Ag_base = Ag.template leftCols<6>();
    auto Ag_base_inv = ocs2::computeFloatingBaseCentroidalMomentumMatrixInverse(Ag_base);

    // TODO: Is this calculation correct?
    vector_t baseAcceleration =
        Ag_base_inv * (h_dot - Ag_dot.template leftCols<6>() * v.head<6>());  // - A_j * q_joint_ddot
    vector3_t baseLinearAcceleration = R_base_world * baseAcceleration.head<3>();
    obsDesiredBaseLinearAcceleration_ = baseLinearAcceleration;  // base linear acceleration expressed in the base frame

    vector3_t eulerZyxAcceleration = baseAcceleration.segment<3>(3);  // euler zyx acceleration
    vector3_t baseAngularAcceleration =
        R_base_world * ocs2::getGlobalAngularAccelerationFromEulerAnglesZyxDerivatives(
                           desiredBaseEulerAngles, desiredEulerAngleDerivatives, eulerZyxAcceleration);
    obsDesiredBaseAngularAcceleration_ =
        baseAngularAcceleration;  // base angular acceleration expressed in the base frame
}

void DtcController::initializeObservations() {
    obsCommand_ = vector_t().setZero(3);
    obsLinearVelocity_ = vector_t().setZero(3);
    obsAngularVelocity_ = vector_t().setZero(3);
    obsProjectedGravity_ = vector_t().setZero(3);
    obsJointPositions_ = vector_t().setZero(12);
    obsJointVelocities_ = vector_t().setZero(12);
    obsPastAction_ = vector_t().setZero(12);
    obsDesiredContacts_ = vector_t().setZero(4);
    obsTimeLeftInPhase_ = vector_t().setZero(4);
    obsDesiredFootholds_ = vector_t().setZero(8);
    obsDesiredJointAngles_ = vector_t().setZero(12);
    obsCurrentDesiredJointAngles_ = vector_t().setZero(12);

    obsDesiredBasePosition_ = vector_t().setZero(3);
    obsDesiredBaseOrientation_ = vector_t().setZero(4);  // this is a quaternion
    obsDesiredBaseLinearVelocity_ = vector_t().setZero(3);
    obsDesiredBaseAngularVelocity_ = vector_t().setZero(3);
    obsDesiredBaseLinearAcceleration_ = vector_t().setZero(3);
    obsDesiredBaseAngularAcceleration_ = vector_t().setZero(3);
}

}  // namespace dtc
}  // namespace tbai
