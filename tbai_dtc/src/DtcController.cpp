// clang-format off
#include <pinocchio/fwd.hpp>
// clang-format on

#include <tbai_core/config/YamlConfig.hpp>
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
#include <tbai_core/Throws.hpp>
#include <vector>

#include <tbai_core/Utils.hpp>

#define DTC_PRINT(message) std::cout << "[Dtc controller] | " << message << std::endl

namespace tbai {

namespace dtc {

DtcController::DtcController(const std::shared_ptr<tbai::core::StateSubscriber> &stateSubscriber)
    : stateSubscriberPtr_(stateSubscriber), mrt_("legged_robot") {
    // Load initial time - the epoch
    initTime_ = tbai::core::getEpochStart();

    // Launch ROS nodes
    ros::NodeHandle nh;
    mrt_.launchNodes(nh);

    // Load parameters
    std::string taskFile, urdfFile, referenceFile;
    TBAI_STD_THROW_IF(!nh.getParam("taskFile", taskFile), "Task file not found");
    TBAI_STD_THROW_IF(!nh.getParam("urdfFile", urdfFile), "URDF file not found");
    TBAI_STD_THROW_IF(!nh.getParam("referenceFile", referenceFile), "Reference file not found");

    // Load default joint angles
    DTC_PRINT("[DtcController] Loading default joint angles");
    defaultJointAngles_ = tbai::core::fromRosConfig<vector_t>("static_controller/stand_controller/joint_angles");

    // Load joint names
    DTC_PRINT("[DtcController] Loading joint names");
    jointNames_ = tbai::core::fromRosConfig<std::vector<std::string>>("joint_names");

    // Load DTC model
    std::string dtcModelFile;
    TBAI_STD_THROW_IF(!nh.getParam("dtcModelFile", dtcModelFile), "DTC model file not found");
    try {
        DTC_PRINT("Loading torch model");
        dtcModel_ = torch::jit::load(dtcModelFile);
    } catch (const c10::Error &e) {
        std::cerr << "Could not load model from: " << dtcModelFile << std::endl;
        throw std::runtime_error("Could not load model");
    }
    DTC_PRINT("Torch model loaded");

    // Do basic ff check
    torch::Tensor input = torch::empty({MODEL_INPUT_SIZE});
    for (int i = 0; i < MODEL_INPUT_SIZE; ++i) input[i] = static_cast<float>(i);
    torch::Tensor output = dtcModel_.forward({input.view({1, -1})}).toTensor().view({-1});
    std::cout << "Basic forward pass check...";
    std::cout << "Input: " << input.view({1, -1}) << std::endl;
    std::cout << "Output: " << output.view({1, -1}) << std::endl;

    DTC_PRINT("Setting up legged interface");
    interfacePtr_ = std::make_unique<LeggedRobotInterface>(taskFile, urdfFile, referenceFile);
    auto &interface = *interfacePtr_;

    DTC_PRINT("Setting up pinocchio interface");
    pinocchioInterfacePtr_ = std::make_unique<PinocchioInterface>(interface.getPinocchioInterface());
    auto &pinocchioInterface = *pinocchioInterfacePtr_;

    DTC_PRINT("Setting up centroidal model mapping");
    centroidalModelMappingPtr_ = std::make_unique<CentroidalModelPinocchioMapping>(interface.getCentroidalModelInfo());
    auto &pinocchioMapping = *centroidalModelMappingPtr_;
    pinocchioMapping.setPinocchioInterface(pinocchioInterface);

    DTC_PRINT("Setting up end effector kinematics");
    endEffectorKinematicsPtr_ = std::make_unique<PinocchioEndEffectorKinematics>(
        pinocchioInterface, *centroidalModelMappingPtr_, interface.modelSettings().contactNames3DoF);
    auto &endEffectorKinematics = *endEffectorKinematicsPtr_;
    endEffectorKinematics.setPinocchioInterface(pinocchioInterface);

    DTC_PRINT("Setting up RBD conversions");
    PinocchioInterface pinint_temp(interface.getPinocchioInterface());
    centroidalModelRbdConversionsPtr_ =
        std::make_unique<CentroidalModelRbdConversions>(pinint_temp, interface.getCentroidalModelInfo());

    DTC_PRINT("Setting up reference velocity generator");
    refVelGen_ = tbai::reference::getReferenceVelocityGeneratorUnique(nh);
    refPub_ = nh.advertise<ocs2_msgs::mpc_target_trajectories>("legged_robot_mpc_target", 1, false);

    DTC_PRINT("Setting up visualizer");
    CentroidalModelPinocchioMapping cmpmVisualizer(interface.getCentroidalModelInfo());
    PinocchioEndEffectorKinematics eeVisualizer(interface.getPinocchioInterface(), cmpmVisualizer,
                                                interface.modelSettings().contactNames3DoF);
    visualizerPtr_ = std::make_unique<LeggedRobotVisualizer>(
        interface.getPinocchioInterface(), interface.getCentroidalModelInfo(), endEffectorKinematics, nh);

    horizon_ = interface.mpcSettings().timeHorizon_;
    // mpcRate_ = interface.mpcSettings().mpcDesiredFrequency_;
    mpcRate_ = 20.0;
    pastAction_ = vector_t().setZero(12);
    DTC_PRINT("Initialization done");
}

void DtcController::publishReference(const TargetTrajectories &targetTrajectories) {
    const auto mpcTargetTrajectoriesMsg = ocs2::ros_msg_conversions::createTargetTrajectoriesMsg(targetTrajectories);
    refPub_.publish(mpcTargetTrajectoriesMsg);
}

tbai_msgs::JointCommandArray DtcController::getCommandMessage(scalar_t currentTime, scalar_t dt) {
    mrt_.spinMRT();
    mrt_.updatePolicy();

    vector3_t linearVelocityObservation = getLinearVelocityObservation(currentTime, dt);
    vector3_t angularVelocityObservation = getAngularVelocityObservation(currentTime, dt);
    vector3_t projectedGravityObservation = getProjectedGravityObservation(currentTime, dt);
    vector3_t commandObservation = getCommandObservation(currentTime, dt);
    vector_t dofPosObservation = getDofPosObservation(currentTime, dt);
    vector_t dofVelObservation = getDofVelObservation(currentTime, dt);
    vector_t pastActionObservation = getPastActionObservation(currentTime, dt);
    vector_t planarFootholdsObservation = getPlanarFootholdsObservation(currentTime, dt);
    vector_t desiredJointAnglesObservation = getDesiredJointAnglesObservation(currentTime, dt);
    vector_t currentDesiredJointAnglesObservation = getCurrentDesiredJointAnglesObservation(currentTime, dt);
    vector_t desiredContactsObservation = getDesiredContactsObservation(currentTime, dt);
    vector_t timeLeftInPhaseObservation = getTimeLeftInPhaseObservation(currentTime, dt);
    vector_t desiredBasePosObservation = getDesiredBasePosObservation(currentTime, dt);
    vector_t orientationDiffObservation = getOrientationDiffObservation(currentTime, dt);
    vector_t desiredBaseLinVelObservation = getDesiredBaseLinVelObservation(currentTime, dt);
    vector_t desiredBaseAngVelObservation = getDesiredBaseAngVelObservation(currentTime, dt);
    vector_t desiredBaseLinAccObservation = getDesiredBaseLinAccObservation(currentTime, dt);
    vector_t desiredBaseAngAccObservation = getDesiredBaseAngAccObservation(currentTime, dt);
    vector_t cpgObservation = getCpgObservation(currentTime, dt);
    vector_t desiredFootPositionsObservation = getDesiredFootPositionsObservation(currentTime, dt);
    vector_t desiredFootVelocitiesObservation = getDesiredFootVelocitiesObservation(currentTime, dt);

    vector_t eigenObservation = vector_t(MODEL_INPUT_SIZE);
    eigenObservation << linearVelocityObservation, angularVelocityObservation, projectedGravityObservation,
        commandObservation, dofPosObservation, dofVelObservation, pastActionObservation, planarFootholdsObservation,
        desiredJointAnglesObservation, currentDesiredJointAnglesObservation, desiredContactsObservation,
        timeLeftInPhaseObservation, desiredBasePosObservation, orientationDiffObservation, desiredBaseLinVelObservation,
        desiredBaseAngVelObservation, desiredBaseLinAccObservation, desiredBaseAngAccObservation, cpgObservation,
        desiredFootPositionsObservation, desiredFootVelocitiesObservation;

    torch::Tensor torchObservation = vector2torch(eigenObservation).view({1, -1});
    torch::Tensor torchAction = dtcModel_.forward({torchObservation}).toTensor().view({-1});

    pastAction_ = torch2vector(torchAction);

    vector_t commandedJointAngles = defaultJointAngles_ + pastAction_ * ACTION_SCALE;

    tbai_msgs::JointCommandArray jointCommandArray;
    jointCommandArray.joint_commands.resize(jointNames_.size());
    for (int i = 0; i < jointNames_.size(); ++i) {
        auto &command = jointCommandArray.joint_commands[i];
        command.joint_name = jointNames_[i];
        command.desired_position = commandedJointAngles[i];
        command.desired_velocity = 0.0;
        command.torque_ff = 0.0;
        command.kp = 80;
        command.kd = 2;
    }

    timeSinceLastMpcUpdate_ += dt;
    if (timeSinceLastMpcUpdate_ > 1.0 / mpcRate_) {
        setObservation();
        publishReference(generateTargetTrajectories(currentTime, dt, commandObservation));
        timeSinceLastMpcUpdate_ = 0.0;
    }

    return jointCommandArray;
}

contact_flag_t DtcController::getDesiredContactFlags(scalar_t currentTime, scalar_t dt) {
    auto &solution = mrt_.getPolicy();
    auto &modeSchedule = solution.modeSchedule_;
    size_t mode = modeSchedule.modeAtTime(currentTime);
    return ocs2::legged_robot::modeNumber2StanceLeg(mode);
}

vector_t DtcController::getTimeLeftInPhase(scalar_t currentTime, scalar_t dt) {
    auto &solution = mrt_.getPolicy();
    auto &modeSchedule = solution.modeSchedule_;

    contact_flag_t desiredContacts = getDesiredContactFlags(currentTime, dt);
    auto contactPhases = ocs2::legged_robot::getContactPhasePerLeg(currentTime, modeSchedule);
    auto swingPhases = ocs2::legged_robot::getSwingPhasePerLeg(currentTime, modeSchedule);

    vector_t timeLeftInPhase(4);
    for (int i = 0; i < 4; ++i) {
        if (desiredContacts[i]) {
            timeLeftInPhase(i) = (1.0 - contactPhases[i].phase) * contactPhases[i].duration;
        } else {
            timeLeftInPhase(i) = (1.0 - swingPhases[i].phase) * swingPhases[i].duration;
        }

        // Make sure there are no nans
        if (std::isnan(timeLeftInPhase(i))) {
            timeLeftInPhase(i) = 0.0;
        }
    }

    return timeLeftInPhase;
}

TargetTrajectories DtcController::generateTargetTrajectories(scalar_t currentTime, scalar_t dt,
                                                             const vector3_t &command) {
    const scalar_t finalTime = currentTime + horizon_;

    SystemObservation sysobs = generateSystemObservation();
    vector_t currentState = sysobs.state;
    vector_t initialState = currentState;

    // Division because it's from the observation vector
    const scalar_t v_x = command(0) / LIN_VEL_SCALE;
    const scalar_t v_y = command(1) / LIN_VEL_SCALE;
    const scalar_t w_z = command(2) / ANG_VEL_SCALE;

    auto &interface = *interfacePtr_;
    auto &centroidalModelInfo = interface.getCentroidalModelInfo();

    scalar_array_t timeTrajectory;
    vector_array_t stateTrajectory;
    vector_array_t inputTrajectory;

    // Insert initial time, state and input
    timeTrajectory.push_back(currentTime);
    stateTrajectory.push_back(initialState);
    inputTrajectory.push_back(vector_t().setZero(24));

    // Complete the rest of the trajectory
    constexpr scalar_t timeStep = 0.1;
    scalar_t time = currentTime;
    vector_t nextState = initialState;

    nextState.tail<12>() = interface.getInitialState().tail<12>();  // Joint angles
    nextState.segment<6>(0) = currentState.segment<6>(0);           // Normalized linear and angular momentum
    nextState.segment<6>(6) = currentState.segment<6>(6);           // World position and orientation
    nextState(8) = 0.54;                                            // z position
    nextState(10) = 0.0;                                            // roll
    nextState(11) = 0.0;                                            // pitch
    nextState(2) = 0.0;                                             // z velocity

    while (time < finalTime) {
        time += timeStep;

        const scalar_t yaw = nextState(9);
        const scalar_t cy = std::cos(yaw);
        const scalar_t sy = std::sin(yaw);

        const scalar_t dx = (cy * v_x - sy * v_y) * timeStep;
        const scalar_t dy = (sy * v_x + cy * v_y) * timeStep;
        const scalar_t dw = w_z * timeStep;

        nextState(0) = dx / timeStep;
        nextState(1) = dy / timeStep;

        nextState(6) += dx;
        nextState(7) += dy;
        nextState(9) += dw;

        timeTrajectory.push_back(time);
        stateTrajectory.push_back(nextState);
        inputTrajectory.push_back(vector_t().setZero(24));
    }

    return TargetTrajectories(timeTrajectory, stateTrajectory, inputTrajectory);
}

std::vector<vector3_t> DtcController::getCurrentFeetPositions(scalar_t currentTime, scalar_t dt) {
    SystemObservation sysobs = generateSystemObservation();
    vector_t currentState = sysobs.state;
    auto &pinocchioMapping = *centroidalModelMappingPtr_;
    auto &interface = *pinocchioInterfacePtr_;
    auto qPinocchio = pinocchioMapping.getPinocchioJointPosition(currentState);
    const auto &model = interface.getModel();
    auto &data = interface.getData();
    pinocchio::forwardKinematics(model, data, qPinocchio);
    pinocchio::updateFramePlacements(model, data);
    auto &endEffector = *endEffectorKinematicsPtr_;
    auto positions = endEffector.getPosition(vector_t());
    return positions;
}

std::vector<vector3_t> DtcController::getCurrentFeetVelocities(scalar_t currentTime, scalar_t dt) {
    SystemObservation sysobs = generateSystemObservation();
    vector_t currentState = sysobs.state;
    vector_t currentInput = sysobs.input;
    auto &pinocchioInterface = *pinocchioInterfacePtr_;
    auto &pinocchioMapping = *centroidalModelMappingPtr_;
    auto &modelInfo = pinocchioMapping.getCentroidalModelInfo();
    vector_t qPinocchio = pinocchioMapping.getPinocchioJointPosition(currentState);
    ocs2::updateCentroidalDynamics(pinocchioInterface, modelInfo, qPinocchio);
    vector_t vPinocchio = pinocchioMapping.getPinocchioJointVelocity(currentState, currentInput);

    const auto &model = pinocchioInterface.getModel();
    auto &data = pinocchioInterface.getData();

    pinocchio::forwardKinematics(model, data, qPinocchio, vPinocchio);
    pinocchio::updateFramePlacements(model, data);

    auto &endEffector = *endEffectorKinematicsPtr_;
    auto velocities = endEffector.getVelocity(vector_t(), vector_t());
    return velocities;
}

std::vector<vector3_t> DtcController::getDesiredFeetPositions(scalar_t currentTime, scalar_t dt) {
    auto &solution = mrt_.getPolicy();
    vector_t optimizedState =
        LinearInterpolation::interpolate(currentTime, solution.timeTrajectory_, solution.stateTrajectory_);
    auto &pinocchioMapping = *centroidalModelMappingPtr_;
    auto &interface = *pinocchioInterfacePtr_;
    auto qPinocchio = pinocchioMapping.getPinocchioJointPosition(optimizedState);
    const auto &model = interface.getModel();
    auto &data = interface.getData();
    pinocchio::forwardKinematics(model, data, qPinocchio);
    pinocchio::updateFramePlacements(model, data);
    auto &endEffector = *endEffectorKinematicsPtr_;
    auto positions = endEffector.getPosition(vector_t());
    return positions;
}

std::vector<vector3_t> DtcController::getDesiredFeetVelocities(scalar_t currentTime, scalar_t dt) {
    auto &solution = mrt_.getPolicy();
    vector_t optimizedState =
        LinearInterpolation::interpolate(currentTime, solution.timeTrajectory_, solution.stateTrajectory_);
    vector_t optimizedInput =
        LinearInterpolation::interpolate(currentTime, solution.timeTrajectory_, solution.inputTrajectory_);
    auto &pinocchioInterface = *pinocchioInterfacePtr_;
    auto &pinocchioMapping = *centroidalModelMappingPtr_;
    auto &modelInfo = pinocchioMapping.getCentroidalModelInfo();
    vector_t qPinocchio = pinocchioMapping.getPinocchioJointPosition(optimizedState);
    ocs2::updateCentroidalDynamics(pinocchioInterface, modelInfo, qPinocchio);
    vector_t vPinocchio = pinocchioMapping.getPinocchioJointVelocity(optimizedState, optimizedInput);

    const auto &model = pinocchioInterface.getModel();
    auto &data = pinocchioInterface.getData();

    pinocchio::forwardKinematics(model, data, qPinocchio, vPinocchio);
    pinocchio::updateFramePlacements(model, data);

    auto &endEffector = *endEffectorKinematicsPtr_;
    auto velocities = endEffector.getVelocity(vector_t(), vector_t());
    return velocities;
}

void DtcController::computeBaseKinematicsAndDynamics(scalar_t currentTime, scalar_t dt, vector3_t &basePos,
                                                     vector3_t &baseOrientation, vector3_t &baseLinearVelocity,
                                                     vector3_t &baseAngularVelocity, vector3_t &baseLinearAcceleration,
                                                     vector3_t &baseAngularAcceleration) {
    auto &solution = mrt_.getPolicy();

    // compute desired MPC state and input
    vector_t desiredState = LinearInterpolation::interpolate(currentTime + ISAAC_SIM_DT, solution.timeTrajectory_,
                                                             solution.stateTrajectory_);
    vector_t desiredInput = LinearInterpolation::interpolate(currentTime + ISAAC_SIM_DT, solution.timeTrajectory_,
                                                             solution.inputTrajectory_);

    // Calculate desired base kinematics and dynamics
    using Vector6 = Eigen::Matrix<scalar_t, 6, 1>;
    Vector6 basePose, baseVelocity, baseAcceleration;
    vector_t jointAccelerations = vector_t::Zero(12);

    auto &centroidalModelRbdConversions = *centroidalModelRbdConversionsPtr_;
    centroidalModelRbdConversions.computeBaseKinematicsFromCentroidalModel(
        desiredState, desiredInput, jointAccelerations, basePose, baseVelocity, baseAcceleration);

    // Unpack data
    vector3_t desiredBasePosition = basePose.head<3>();
    vector3_t desiredBaseOrientation = basePose.tail<3>();  // zyx euler angles

    vector3_t desiredBaseLinearVelocity = baseVelocity.head<3>();
    vector3_t desiredBaseAngularVelocity = baseVelocity.tail<3>();

    vector3_t desiredBaseLinearAcceleration = baseAcceleration.head<3>();
    vector3_t desiredBaseAngularAcceleration = baseAcceleration.tail<3>();

    basePos = desiredBasePosition;
    baseOrientation = desiredBaseOrientation;
    baseLinearVelocity = desiredBaseLinearVelocity;
    baseAngularVelocity = desiredBaseAngularVelocity;
    baseLinearAcceleration = desiredBaseLinearAcceleration;
    baseAngularAcceleration = desiredBaseAngularAcceleration;
}

vector3_t DtcController::getLinearVelocityObservation(scalar_t currentTime, scalar_t dt) const {
    const vector_t &rbdState = stateSubscriberPtr_->getLatestRbdState();
    return rbdState.segment<3>(9) * LIN_VEL_SCALE;  // COM velocity - already expessed in base frame
}

vector3_t DtcController::getAngularVelocityObservation(scalar_t currentTime, scalar_t dt) const {
    const vector_t &rbdState = stateSubscriberPtr_->getLatestRbdState();
    return rbdState.segment<3>(6) * ANG_VEL_SCALE;  // Angular velocity - already expessed in base frame
}

vector3_t DtcController::getProjectedGravityObservation(scalar_t currentTime, scalar_t dt) const {
    const vector_t &rbdState = stateSubscriberPtr_->getLatestRbdState();
    const matrix3_t R_base_world = getRotationMatrixBaseWorld(rbdState);
    return R_base_world * (vector3_t() << 0.0, 0.0, -1.0).finished() * GRAVITY_SCALE;
}

vector3_t DtcController::getCommandObservation(scalar_t currentTime, scalar_t dt) {
    tbai::reference::ReferenceVelocity refvel = refVelGen_->getReferenceVelocity(currentTime, 0.1);
    return vector3_t(refvel.velocity_x * LIN_VEL_SCALE, refvel.velocity_y * LIN_VEL_SCALE,
                     refvel.yaw_rate * ANG_VEL_SCALE);
}

vector_t DtcController::getDofPosObservation(scalar_t currentTime, scalar_t dt) const {
    const vector_t &rbdState = stateSubscriberPtr_->getLatestRbdState();
    const vector_t jointAngles = rbdState.segment<12>(12);
    return (jointAngles - defaultJointAngles_) * DOF_POS_SCALE;
}

vector_t DtcController::getDofVelObservation(scalar_t currentTime, scalar_t dt) const {
    const vector_t &rbdState = stateSubscriberPtr_->getLatestRbdState();
    const vector_t jointVelocities = rbdState.segment<12>(24);
    return jointVelocities * DOF_VEL_SCALE;
}

vector_t DtcController::getPastActionObservation(scalar_t currentTime, scalar_t dt) const {
    return pastAction_ * PAST_ACTION_SCALE;
}

vector_t DtcController::getPlanarFootholdsObservation(scalar_t currentTime, scalar_t dt) {
    auto &solution = mrt_.getPolicy();
    auto &modeSchedule = solution.modeSchedule_;
    auto timeLeftInPhase = getTimeLeftInPhase(currentTime, dt);

    std::vector<vector3_t> feetPositions = getCurrentFeetPositions(currentTime, dt);
    const auto &rbdState = stateSubscriberPtr_->getLatestRbdState();
    vector_t out = vector_t().setZero(8);

    for (int j = 0; j < 4; ++j) {
        const scalar_t timeLeft = timeLeftInPhase(j);

        auto optimizedState =
            LinearInterpolation::interpolate(timeLeft, solution.timeTrajectory_, solution.stateTrajectory_);

        // Compute forward kinematics
        auto &pinocchioMapping = *centroidalModelMappingPtr_;
        auto &interface = *pinocchioInterfacePtr_;
        auto qPinocchio = pinocchioMapping.getPinocchioJointPosition(optimizedState);
        const auto &model = interface.getModel();
        auto &data = interface.getData();
        pinocchio::forwardKinematics(model, data, qPinocchio);
        pinocchio::updateFramePlacements(model, data);

        // Update end effector kinematics
        auto &endEffector = *endEffectorKinematicsPtr_;
        auto positions = endEffector.getPosition(vector_t());

        // Update desired footholds
        matrix3_t R_base_world_yaw = getRotationMatrixBaseWorldYaw(rbdState);
        vector_t footholdDesired = positions[j];
        vector_t footholdCurrent = feetPositions[j];

        // This section here is different from env.py
        // footholdInBase(2) = 0.0; // TODO: Is this necessary? I don't think so
        // footholdCurrent(2) = 0.0; // TODO: Is this necessary? I don't think so
        vector_t footholdInBase = R_base_world_yaw * (footholdDesired - footholdCurrent);

        out.segment<2>(2 * j) = footholdInBase.head<2>();
    }

    return out;
}

vector_t DtcController::getDesiredJointAnglesObservation(scalar_t currentTime, scalar_t dt) {
    auto &solution = mrt_.getPolicy();
    auto &modeSchedule = solution.modeSchedule_;
    auto timeLeftInPhase = getTimeLeftInPhase(currentTime, dt);
    vector_t out = vector_t().setZero(12);

    for (int j = 0; j < 4; ++j) {
        const scalar_t timeLeft = timeLeftInPhase(j);
        auto optimizedState =
            LinearInterpolation::interpolate(timeLeft, solution.timeTrajectory_, solution.stateTrajectory_);
        auto optimizedJointAngles = optimizedState.segment<12>(12);
        out.segment<3>(3 * j) = optimizedJointAngles.segment<3>(3 * j);
    }

    // Subtract default joint angles - this is different from env.py
    out -= defaultJointAngles_;

    return out;
}

vector_t DtcController::getCurrentDesiredJointAnglesObservation(scalar_t currentTime, scalar_t dt) {
    auto &solution = mrt_.getPolicy();
    auto optimizedState = LinearInterpolation::interpolate(currentTime + ISAAC_SIM_DT, solution.timeTrajectory_,
                                                           solution.stateTrajectory_);
    auto optimizedJointAngles = optimizedState.segment<12>(12);
    vector_t out = optimizedJointAngles - defaultJointAngles_;
    return out;
}

vector_t DtcController::getDesiredContactsObservation(scalar_t currentTime, scalar_t dt) {
    auto desiredContacts = getDesiredContactFlags(currentTime, dt);
    vector_t out = vector_t().setZero(4);
    for (int i = 0; i < 4; ++i) {
        out(i) = static_cast<scalar_t>(desiredContacts[i]);
    }
    return out;
}

vector_t DtcController::getTimeLeftInPhaseObservation(scalar_t currentTime, scalar_t dt) {
    return getTimeLeftInPhase(currentTime, dt);
}

vector_t DtcController::getDesiredBasePosObservation(scalar_t currentTime, scalar_t dt) {
    vector3_t basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity, baseLinearAcceleration,
        baseAngularAcceleration;
    computeBaseKinematicsAndDynamics(currentTime, dt, basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity,
                                     baseLinearAcceleration, baseAngularAcceleration);
    const auto &rbdState = stateSubscriberPtr_->getLatestRbdState();
    vector3_t basePosCurrent = rbdState.segment<3>(3);
    vector3_t basePosDesired = basePos;
    matrix3_t R_base_world_yaw = getRotationMatrixBaseWorldYaw(rbdState);
    vector3_t out = R_base_world_yaw * (basePosDesired - basePosCurrent);
    return out;
}

vector_t DtcController::getOrientationDiffObservation(scalar_t currentTime, scalar_t dt) {
    vector3_t basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity, baseLinearAcceleration,
        baseAngularAcceleration;
    computeBaseKinematicsAndDynamics(currentTime, dt, basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity,
                                     baseLinearAcceleration, baseAngularAcceleration);
    const auto &rbdState = stateSubscriberPtr_->getLatestRbdState();

    vector3_t eulerAnglesZyxCurrent = getOcs2ZyxEulerAngles(rbdState);
    vector3_t eulerAnglesZyxDesired = baseOrientation;

    // Convert euler angles to quaternions
    quaternion_t quatCurrent = this->getQuaternionFromEulerAnglesZyx(eulerAnglesZyxCurrent);
    quaternion_t quatDesired = this->getQuaternionFromEulerAnglesZyx(eulerAnglesZyxDesired);

    // Invert current quaternion
    quaternion_t quatCurrentInverse = quatCurrent.conjugate();

    // Compute the difference
    quaternion_t quatDiff = quatDesired * quatCurrentInverse;

    const scalar_t x = quatDiff.x();
    const scalar_t y = quatDiff.y();
    const scalar_t z = quatDiff.z();
    const scalar_t w = quatDiff.w();

    vector_t orientationDiff = (vector_t(4) << x, y, z, w).finished();

    return orientationDiff;
}

vector_t DtcController::getDesiredBaseLinVelObservation(scalar_t currentTime, scalar_t dt) {
    vector3_t basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity, baseLinearAcceleration,
        baseAngularAcceleration;
    computeBaseKinematicsAndDynamics(currentTime, dt, basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity,
                                     baseLinearAcceleration, baseAngularAcceleration);
    const auto &rbdState = stateSubscriberPtr_->getLatestRbdState();
    matrix3_t R_base_world = getRotationMatrixBaseWorld(rbdState);
    vector3_t baseLinVelDesiredWorld = baseLinearVelocity;
    vector3_t baseLinVelDesiredBase = R_base_world * baseLinVelDesiredWorld;
    return baseLinVelDesiredBase;
}

vector_t DtcController::getDesiredBaseAngVelObservation(scalar_t currentTime, scalar_t dt) {
    vector3_t basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity, baseLinearAcceleration,
        baseAngularAcceleration;
    computeBaseKinematicsAndDynamics(currentTime, dt, basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity,
                                     baseLinearAcceleration, baseAngularAcceleration);
    const auto &rbdState = stateSubscriberPtr_->getLatestRbdState();
    matrix3_t R_base_world = getRotationMatrixBaseWorld(rbdState);
    vector3_t baseAngVelDesiredWorld = baseAngularVelocity;
    vector3_t baseAngVelDesiredBase = R_base_world * baseAngVelDesiredWorld;
    return baseAngVelDesiredBase;
}

vector_t DtcController::getDesiredBaseLinAccObservation(scalar_t currentTime, scalar_t dt) {
    vector3_t basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity, baseLinearAcceleration,
        baseAngularAcceleration;
    computeBaseKinematicsAndDynamics(currentTime, dt, basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity,
                                     baseLinearAcceleration, baseAngularAcceleration);
    const auto &rbdState = stateSubscriberPtr_->getLatestRbdState();
    matrix3_t R_base_world = getRotationMatrixBaseWorld(rbdState);
    vector3_t baseLinAccDesiredWorld = baseLinearAcceleration;
    vector3_t baseLinAccDesiredBase = R_base_world * baseLinAccDesiredWorld;
    return baseLinAccDesiredBase;
}

vector_t DtcController::getDesiredBaseAngAccObservation(scalar_t currentTime, scalar_t dt) {
    vector3_t basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity, baseLinearAcceleration,
        baseAngularAcceleration;
    computeBaseKinematicsAndDynamics(currentTime, dt, basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity,
                                     baseLinearAcceleration, baseAngularAcceleration);
    const auto &rbdState = stateSubscriberPtr_->getLatestRbdState();
    matrix3_t R_base_world = getRotationMatrixBaseWorld(rbdState);
    vector3_t baseAngAccDesiredWorld = baseAngularAcceleration;
    vector3_t baseAngAccDesiredBase = R_base_world * baseAngAccDesiredWorld;
    return baseAngAccDesiredBase;
}

vector_t DtcController::getCpgObservation(scalar_t currentTime, scalar_t dt) {
    auto &solution = mrt_.getPolicy();
    auto &modeSchedule = solution.modeSchedule_;

    auto desiredContacts = getDesiredContactFlags(currentTime, dt);

    auto contactPhases = ocs2::legged_robot::getContactPhasePerLeg(currentTime, modeSchedule);
    auto swingPhases = ocs2::legged_robot::getSwingPhasePerLeg(currentTime, modeSchedule);

    vector_t phases = vector_t::Zero(4);

    // LH, RF - phase in [0, PI]
    // LF, RH - phase in [PI, 2*PI]
    // Basically when LF lifts off the phase is 0
    constexpr scalar_t PI = 3.14159265358979323846;
    for (int j = 0; j < 4; ++j) {
        if (desiredContacts[j]) {
            phases(j) = PI + contactPhases[j].phase * PI;
        } else {
            phases(j) = swingPhases[j].phase * PI;
        }

        if (phases(j) > 2 * PI) phases(j) -= 2 * PI;
        if (phases(j) < 0) phases(j) += 2 * PI;
    }

    // Perform swap - LF, RF, LH, RH -> LF, LH, RF, RH
    std::swap(phases(1), phases(2));

    // Compute cpg obs
    vector_t out = vector_t::Zero(8);
    for (int j = 0; j < 4; ++j) {
        const scalar_t phase = phases(j);
        const scalar_t c = std::cos(phase);
        const scalar_t s = std::sin(phase);
        out(j) = c;
        out(j + 4) = s;
    }

    return out;
}

vector_t DtcController::getDesiredFootPositionsObservation(scalar_t currentTime, scalar_t dt) {
    auto desiredFootPositions = getDesiredFeetPositions(currentTime, dt);
    auto currentFootPositions = getCurrentFeetPositions(currentTime, dt);

    matrix3_t R_base_world_yaw = getRotationMatrixBaseWorldYaw(stateSubscriberPtr_->getLatestRbdState());

    vector_t lf_pos = -R_base_world_yaw * (desiredFootPositions[0] - currentFootPositions[0]);
    vector_t rf_pos = -R_base_world_yaw * (desiredFootPositions[1] - currentFootPositions[1]);
    vector_t lh_pos = -R_base_world_yaw * (desiredFootPositions[2] - currentFootPositions[2]);
    vector_t rh_pos = -R_base_world_yaw * (desiredFootPositions[3] - currentFootPositions[3]);

    vector_t out(4 * 3);
    out << lf_pos, lh_pos, rf_pos, rh_pos;
    return out;
}
vector_t DtcController::getDesiredFootVelocitiesObservation(scalar_t currentTime, scalar_t dt) {
    auto desiredFootVelocities = getDesiredFeetVelocities(currentTime, dt);
    auto currentFootVelocities = getCurrentFeetVelocities(currentTime, dt);

    matrix3_t R_base_world_yaw = getRotationMatrixBaseWorldYaw(stateSubscriberPtr_->getLatestRbdState());

    vector_t lf_vel = -R_base_world_yaw * (desiredFootVelocities[0] - currentFootVelocities[0]);
    vector_t rf_vel = -R_base_world_yaw * (desiredFootVelocities[1] - currentFootVelocities[1]);
    vector_t lh_vel = -R_base_world_yaw * (desiredFootVelocities[2] - currentFootVelocities[2]);
    vector_t rh_vel = -R_base_world_yaw * (desiredFootVelocities[3] - currentFootVelocities[3]);

    vector_t out(4 * 3);
    out << lf_vel, lh_vel, rf_vel, rh_vel;
    return out;
}

void DtcController::visualize() {
    mrt_.spinMRT();
    mrt_.updatePolicy();
    visualizerPtr_->update(generateSystemObservation(), mrt_.getPolicy(), mrt_.getCommand());
}

void DtcController::changeController(const std::string &controllerType, scalar_t currentTime) {
    resetMpc();
}

bool DtcController::isSupported(const std::string &controllerType) {
    return controllerType == "DTC";
}

ocs2::SystemObservation DtcController::generateSystemObservation() {
    // Unpack latest rbc state
    const vector_t &rbdState = stateSubscriberPtr_->getLatestRbdState();
    const scalar_t observationTime = stateSubscriberPtr_->getLatestRbdStamp().toSec() - initTime_;
    const size_t mode = ocs2::legged_robot::stanceLeg2ModeNumber(stateSubscriberPtr_->getContactFlags());

    const matrix3_t R_world_base = getRotationMatrixWorldBase(rbdState);  // TODO: room for optimization
    const matrix3_t R_base_world = getRotationMatrixBaseWorld(rbdState);  // TODO: room for optimization
    const vector3_t eulerAnglesZyx = getOcs2ZyxEulerAngles(rbdState);     // TOD: room for optimization
    const vector_t jointAngles = rbdState.segment<12>(12);
    const vector_t jointVelocities = rbdState.segment<12>(24);

    // Set observation time
    ocs2::SystemObservation observation;
    observation.time = observationTime;
    observation.mode = mode;
    observation.input = vector_t().setZero(24);
    observation.input.tail<12>() = jointVelocities;

    // 3 normalized linear momentum, 3 normalized angular momentum, 3 com position, 3 ypr, 12 joint angles
    observation.state = vector_t().setZero(3 + 3 + 3 + 3 + 12);
    observation.state.segment<3>(0) = R_world_base * rbdState.segment<3>(9);  // v_com in world frame
    observation.state.segment<3>(3) = vector_t().setZero(3);                  // normalized angular momentum (not used)
    observation.state.segment<3>(6) = rbdState.segment<3>(3);                 // com position
    observation.state.segment<3>(9) = eulerAnglesZyx;                         // ypr
    observation.state.segment<12>(12) = jointAngles;                          // LF, LH, RF, RH

    return observation;
}

void DtcController::resetMpc() {
    // Wait to receive observation
    stateSubscriberPtr_->waitTillInitialized();

    // Prepare initial observation for MPC
    ocs2::SystemObservation mpcObservation = generateSystemObservation();

    // Prepare target trajectory
    ocs2::TargetTrajectories initTargetTrajectories({0.0}, {mpcObservation.state}, {mpcObservation.input});
    mrt_.resetMpcNode(initTargetTrajectories);

    while (!mrt_.initialPolicyReceived() && ros::ok()) {
        ROS_INFO("Waiting for initial policy...");
        ros::spinOnce();
        mrt_.spinMRT();
        mrt_.setCurrentObservation(generateSystemObservation());
        ros::Duration(0.1).sleep();
    }

    ROS_INFO("Initial policy received.");
}

void DtcController::setObservation() {
    mrt_.setCurrentObservation(generateSystemObservation());
}

torch::Tensor vector2torch(const vector_t &v) {
    const long rows = static_cast<long>(v.rows());
    auto out = torch::empty({rows});
    float *data = out.data_ptr<float>();

    Eigen::Map<Eigen::VectorXf> map(data, rows);
    map = v.cast<float>();

    return out;
}

torch::Tensor matrix2torch(const matrix_t &m) {
    const long rows = static_cast<long>(m.rows());
    const long cols = static_cast<long>(m.cols());
    auto out = torch::empty({rows, cols});
    float *data = out.data_ptr<float>();

    Eigen::Map<Eigen::MatrixXf> map(data, cols, rows);  // "view" it as a [cols x rows] matrix
    map = m.transpose().cast<float>();

    return out;
}

vector_t torch2vector(const torch::Tensor &t) {
    const size_t rows = t.size(0);
    float *t_data = t.data_ptr<float>();

    vector_t out(rows);
    Eigen::Map<Eigen::VectorXf> map(t_data, rows);
    out = map.cast<scalar_t>();
    return out;
}

matrix_t torch2matrix(const torch::Tensor &t) {
    const size_t rows = t.size(0);
    const size_t cols = t.size(1);
    float *t_data = t.data_ptr<float>();

    matrix_t out(cols, rows);
    Eigen::Map<Eigen::MatrixXf> map(t_data, cols, rows);
    out = map.cast<scalar_t>();
    return out.transpose();
}

}  // namespace dtc
}  // namespace tbai
