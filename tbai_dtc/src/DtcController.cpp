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
#include <ocs2_switched_model_interface/core/MotionPhaseDefinition.h>

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
    : stateSubscriberPtr_(stateSubscriber), mrt_("anymal") {
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
    refPub_ = nh.advertise<ocs2_msgs::mpc_target_trajectories>("anymal_mpc_target", 1, false);

    DTC_PRINT("Setting up visualizer");
    CentroidalModelPinocchioMapping cmpmVisualizer(interface.getCentroidalModelInfo());
    PinocchioEndEffectorKinematics eeVisualizer(interface.getPinocchioInterface(), cmpmVisualizer,
                                                interface.modelSettings().contactNames3DoF);
    visualizerPtr_ = std::make_unique<LeggedRobotVisualizer>(
        interface.getPinocchioInterface(), interface.getCentroidalModelInfo(), endEffectorKinematics, nh);

    horizon_ = interface.mpcSettings().timeHorizon_;
    // mpcRate_ = interface.mpcSettings().mpcDesiredFrequency_;
    mpcRate_ = 10.0;
    pastAction_ = vector_t().setZero(12);

    if (!blind_) {
        DTC_PRINT("Initializing gridmap interface");
        gridmap_ = tbai::gridmap::getGridmapInterfaceUnique();
    }

    DTC_PRINT("Initializing quadruped interface");
    std::string urdf, taskFolder;
    ros::param::get("robot_description", urdf);
    ros::param::get("task_folder", taskFolder);
    quadrupedInterface_ = anymal::getAnymalInterface(urdf, taskFolder);
    auto &quadrupedInterface = *quadrupedInterface_;
    comModel_.reset(quadrupedInterface.getComModel().clone());
    kinematicsModel_.reset(quadrupedInterface.getKinematicModel().clone());
    visualizer_.reset(new switched_model::QuadrupedVisualizer(quadrupedInterface.getKinematicModel(),
                                                              quadrupedInterface.getJointNames(),
                                                              quadrupedInterface.getBaseName(), nh));

    DTC_PRINT("Initialization done");
}

void DtcController::publishReference(const TargetTrajectories &targetTrajectories) {
    const auto mpcTargetTrajectoriesMsg = ocs2::ros_msg_conversions::createTargetTrajectoriesMsg(targetTrajectories);
    refPub_.publish(mpcTargetTrajectoriesMsg);
}

tbai_msgs::JointCommandArray DtcController::getCommandMessage(scalar_t currentTime, scalar_t dt) {
    mrt_.spinMRT();
    mrt_.updatePolicy();

    tbai_msgs::JointCommandArray jointCommandArray;
    jointCommandArray.joint_commands.resize(jointNames_.size());
    for (int i = 0; i < jointNames_.size(); ++i) {
        auto &command = jointCommandArray.joint_commands[i];
        command.joint_name = jointNames_[i];
        command.desired_position = defaultJointAngles_[i];
        command.desired_velocity = 0.0;
        command.torque_ff = 0.0;
        command.kp = 400;
        command.kd = 20;
    }

    vector3_t linearVelocityObservation = getLinearVelocityObservation(currentTime, dt);
    vector3_t angularVelocityObservation = getAngularVelocityObservation(currentTime, dt);
    vector3_t projectedGravityObservation = getProjectedGravityObservation(currentTime, dt);
    vector3_t commandObservation = getCommandObservation(currentTime, dt);
    vector_t dofPosObservation = getDofPosObservation(currentTime, dt);
    vector_t dofVelObservation = getDofVelObservation(currentTime, dt);
    vector_t pastActionObservation = getPastActionObservation(currentTime, dt);
    vector_t planarFootholdsObservation = getPlanarFootholdsObservation(currentTime, dt);
    // vector_t desiredJointAnglesObservation = getDesiredJointAnglesObservation(currentTime, dt);
    // vector_t currentDesiredJointAnglesObservation = getCurrentDesiredJointAnglesObservation(currentTime, dt);
    // vector_t desiredContactsObservation = getDesiredContactsObservation(currentTime, dt);
    // vector_t timeLeftInPhaseObservation = getTimeLeftInPhaseObservation(currentTime, dt);
    // vector_t desiredBasePosObservation = getDesiredBasePosObservation(currentTime, dt);
    // vector_t orientationDiffObservation = getOrientationDiffObservation(currentTime, dt);
    // vector_t desiredBaseLinVelObservation = getDesiredBaseLinVelObservation(currentTime, dt);
    // vector_t desiredBaseAngVelObservation = getDesiredBaseAngVelObservation(currentTime, dt);
    // vector_t desiredBaseLinAccObservation = getDesiredBaseLinAccObservation(currentTime, dt);
    // vector_t desiredBaseAngAccObservation = getDesiredBaseAngAccObservation(currentTime, dt);
    // vector_t cpgObservation = getCpgObservation(currentTime, dt);
    // vector_t desiredFootPositionsObservation = getDesiredFootPositionsObservation(currentTime, dt);
    // vector_t desiredFootVelocitiesObservation = getDesiredFootVelocitiesObservation(currentTime, dt);

    // vector_t eigenObservation = vector_t(MODEL_INPUT_SIZE);
    // eigenObservation << linearVelocityObservation, angularVelocityObservation, projectedGravityObservation,
    //     commandObservation, dofPosObservation, dofVelObservation, pastActionObservation, planarFootholdsObservation,
    //     desiredJointAnglesObservation, currentDesiredJointAnglesObservation, desiredContactsObservation,
    //     timeLeftInPhaseObservation, desiredBasePosObservation, orientationDiffObservation,
    //     desiredBaseLinVelObservation, desiredBaseAngVelObservation, desiredBaseLinAccObservation,
    //     desiredBaseAngAccObservation, cpgObservation, desiredFootPositionsObservation,
    //     desiredFootVelocitiesObservation;

    // torch::Tensor torchObservation = vector2torch(eigenObservation).view({1, -1});
    // torch::Tensor torchAction = dtcModel_.forward({torchObservation}).toTensor().view({-1});

    // pastAction_ = torch2vector(torchAction);

    // vector_t commandedJointAngles = defaultJointAngles_ + pastAction_ * ACTION_SCALE;

    // tbai_msgs::JointCommandArray jointCommandArray;
    // jointCommandArray.joint_commands.resize(jointNames_.size());
    // for (int i = 0; i < jointNames_.size(); ++i) {
    //     auto &command = jointCommandArray.joint_commands[i];
    //     command.joint_name = jointNames_[i];
    //     command.desired_position = commandedJointAngles[i];
    //     command.desired_velocity = 0.0;
    //     command.torque_ff = 0.0;
    //     command.kp = 80;
    //     command.kd = 2;
    // }

    timeSinceLastMpcUpdate_ += dt;
    if (timeSinceLastMpcUpdate_ > 1.0 / mpcRate_) {
        setObservation();
        // TODO: Put this into its own thread
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
    auto &eventTimes = modeSchedule.eventTimes;
    auto it = std::lower_bound(eventTimes.begin(), eventTimes.end(), currentTime);
    if (it == eventTimes.end()) {
        return vector_t::Zero(4);
    }
    return vector_t::Ones(4) * (*it - currentTime);
}

TargetTrajectories DtcController::generateTargetTrajectories(scalar_t currentTime, scalar_t dt,
                                                             const vector3_t &command) {
    switched_model::BaseReferenceTrajectory baseReferenceTrajectory =
        generateExtrapolatedBaseReference(getBaseReferenceHorizon(currentTime), getBaseReferenceState(currentTime),
                                          getBaseReferenceCommand(currentTime, command), gridmap_->getMap(), 0.53, 0.3);

    constexpr size_t STATE_DIM = 6 + 6 + 12;
    constexpr size_t INPUT_DIM = 12 + 12;

    // Generate target trajectory
    ocs2::scalar_array_t desiredTimeTrajectory = std::move(baseReferenceTrajectory.time);
    const size_t N = desiredTimeTrajectory.size();
    ocs2::vector_array_t desiredStateTrajectory(N);
    ocs2::vector_array_t desiredInputTrajectory(N, ocs2::vector_t::Zero(INPUT_DIM));
    for (size_t i = 0; i < N; ++i) {
        ocs2::vector_t state = ocs2::vector_t::Zero(STATE_DIM);

        // base orientation
        state.head<3>() = baseReferenceTrajectory.eulerXyz[i];

        auto Rt = switched_model::rotationMatrixOriginToBase(baseReferenceTrajectory.eulerXyz[i]);

        // base position
        state.segment<3>(3) = baseReferenceTrajectory.positionInWorld[i];

        // base angular velocity
        state.segment<3>(6) = Rt * baseReferenceTrajectory.angularVelocityInWorld[i];

        // base linear velocity
        state.segment<3>(9) = Rt * baseReferenceTrajectory.linearVelocityInWorld[i];

        // joint angles
        state.segment<12>(12) = quadrupedInterface_->getInitialState().segment<12>(12);

        desiredStateTrajectory[i] = std::move(state);
    }

    return TargetTrajectories(std::move(desiredTimeTrajectory), std::move(desiredStateTrajectory),
                              std::move(desiredInputTrajectory));
}

std::vector<vector3_t> DtcController::getCurrentFeetPositions(scalar_t currentTime, scalar_t dt) {
    SystemObservation sysobs = generateSystemObservation();
    vector_t currentState = sysobs.state;
    auto &quadcom = *comModel_;
    auto &kin = *kinematicsModel_;
    auto basePoseOcs2 = switched_model::getBasePose(currentState);
    auto jointAnglesOcs2 = switched_model::getJointPositions(currentState);
    std::vector<vector3_t> feetPositions(4);
    for (int legidx = 0; legidx < 4; ++legidx) {
        auto footPosition = kin.footPositionInOriginFrame(legidx, basePoseOcs2, jointAnglesOcs2);
        feetPositions[legidx] = footPosition;
    }
    return feetPositions;
}

std::vector<vector3_t> DtcController::getCurrentFeetVelocities(scalar_t currentTime, scalar_t dt) {
    SystemObservation sysobs = generateSystemObservation();
    vector_t currentState = sysobs.state;
    vector_t currentInput = sysobs.input;
    auto &quadcom = *comModel_;
    auto &kin = *kinematicsModel_;
    auto basePoseOcs2 = switched_model::getBasePose(currentState);
    auto baseTwistOcs2 = switched_model::getBaseLocalVelocities(currentState);
    auto jointAnglesOcs2 = switched_model::getJointPositions(currentState);
    auto jointVelocitiesOcs2 = switched_model::getJointVelocities(currentInput);
    std::vector<vector3_t> feetVelocities(4);
    for (int legidx = 0; legidx < 4; ++legidx) {
        auto footVelocity =
            kin.footVelocityInOriginFrame(legidx, basePoseOcs2, baseTwistOcs2, jointAnglesOcs2, jointVelocitiesOcs2);
        feetVelocities[legidx] = footVelocity;
    }
    return feetVelocities;
}

std::vector<vector3_t> DtcController::getDesiredFeetPositions(scalar_t currentTime, scalar_t dt) {
    auto &solution = mrt_.getPolicy();
    vector_t optimizedState =
        LinearInterpolation::interpolate(currentTime, solution.timeTrajectory_, solution.stateTrajectory_);
    auto &quadcom = *comModel_;
    auto &kin = *kinematicsModel_;
    auto basePoseOcs2 = switched_model::getBasePose(optimizedState);
    auto jointAnglesOcs2 = switched_model::getJointPositions(optimizedState);
    std::vector<vector3_t> feetPositions(4);
    for (int legidx = 0; legidx < 4; ++legidx) {
        auto footPosition = kin.footPositionInOriginFrame(legidx, basePoseOcs2, jointAnglesOcs2);
        feetPositions[legidx] = footPosition;
    }
    return feetPositions;
}

std::vector<vector3_t> DtcController::getDesiredFeetVelocities(scalar_t currentTime, scalar_t dt) {
    auto &solution = mrt_.getPolicy();
    vector_t optimizedState =
        LinearInterpolation::interpolate(currentTime, solution.timeTrajectory_, solution.stateTrajectory_);
    vector_t optimizedInput =
        LinearInterpolation::interpolate(currentTime, solution.timeTrajectory_, solution.inputTrajectory_);
    auto &quadcom = *comModel_;
    auto &kin = *kinematicsModel_;
    auto basePoseOcs2 = switched_model::getBasePose(optimizedState);
    auto baseTwistOcs2 = switched_model::getBaseLocalVelocities(optimizedState);
    auto jointAnglesOcs2 = switched_model::getJointPositions(optimizedState);
    auto jointVelocitiesOcs2 = switched_model::getJointVelocities(optimizedState);  // TODO: Bug, this should be input
    std::vector<vector3_t> feetVelocities(4);
    for (int legidx = 0; legidx < 4; ++legidx) {
        auto footVelocity =
            kin.footVelocityInOriginFrame(legidx, basePoseOcs2, baseTwistOcs2, jointAnglesOcs2, jointVelocitiesOcs2);
        feetVelocities[legidx] = footVelocity;
    }
    return feetVelocities;
}

void DtcController::computeBaseKinematicsAndDynamics(scalar_t currentTime, scalar_t dt, vector3_t &basePos,
                                                     vector_t &baseOrientation, vector3_t &baseLinearVelocity,
                                                     vector3_t &baseAngularVelocity, vector3_t &baseLinearAcceleration,
                                                     vector3_t &baseAngularAcceleration) {
    auto &solution = mrt_.getPolicy();

    // compute desired MPC state and input
    vector_t desiredState = LinearInterpolation::interpolate(currentTime + ISAAC_SIM_DT, solution.timeTrajectory_,
                                                             solution.stateTrajectory_);
    vector_t desiredInput = LinearInterpolation::interpolate(currentTime + ISAAC_SIM_DT, solution.timeTrajectory_,
                                                             solution.inputTrajectory_);

    auto &quadcom = *comModel_;
    auto &kin = *kinematicsModel_;

    auto basePoseOcs2 = switched_model::getBasePose(desiredState);
    auto baseVelocityOcs2 = switched_model::getBaseLocalVelocities(desiredState);

    // Desired base orinetation as a quaternion
    quaternion_t desiredBaseOrientationQuat = ocs2rpy2quat(basePoseOcs2.head<3>());  // ocs2 xyz to quaternion
    matrix3_t rotationWorldBase = desiredBaseOrientationQuat.toRotationMatrix();

    const vector_t &basePosett = desiredState.head<6>();
    const vector_t &baseVelocitytt = desiredState.segment<6>(6);
    const vector_t &jointPositionstt = desiredState.tail<switched_model::JOINT_COORDINATE_SIZE>();
    const vector_t &jointVelocitiestt = desiredInput.tail<switched_model::JOINT_COORDINATE_SIZE>();
    const vector_t &jointAccelerationstt = vector_t::Zero(switched_model::JOINT_COORDINATE_SIZE);

    // forcesOnBaseInBaseFrame = [torque (3); force (3)]
    vector_t forcesOnBaseInBaseFrame = vector_t::Zero(6);
    for (size_t i = 0; i < 4; ++i) {
        // force at foot expressed in base frame
        const vector3_t &forceAtFoot = desiredInput.segment<3>(3 * i);

        // base force
        forcesOnBaseInBaseFrame.tail<3>() += forceAtFoot;

        // base torque
        vector3_t footPosition = kin.positionBaseToFootInBaseFrame(i, jointPositionstt);
        forcesOnBaseInBaseFrame.head<3>() += footPosition.cross(forceAtFoot);
    }

    vector_t baseAccelerationLocal = quadcom.calculateBaseLocalAccelerations(
        basePosett, baseVelocitytt, jointPositionstt, jointVelocitiestt, jointAccelerationstt, forcesOnBaseInBaseFrame);

    // Unpack data
    vector3_t desiredBasePosition = basePoseOcs2.tail<3>();
    vector_t desiredBaseOrientation = desiredBaseOrientationQuat.coeffs();  // zyx euler angles

    vector3_t desiredBaseLinearVelocity = rotationWorldBase * baseVelocityOcs2.tail<3>();
    vector3_t desiredBaseAngularVelocity = rotationWorldBase * baseVelocityOcs2.head<3>();

    vector3_t desiredBaseLinearAcceleration = rotationWorldBase * baseAccelerationLocal.tail<3>();
    vector3_t desiredBaseAngularAcceleration = rotationWorldBase * baseAccelerationLocal.head<3>();

    // Update desired base
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

    std::vector<vector3_t> currentFootPositions = getCurrentFeetPositions(currentTime, dt);
    const auto &rbdState = stateSubscriberPtr_->getLatestRbdState();
    vector_t out = vector_t().setZero(8);

    for (int legidx = 0; legidx < 4; ++legidx) {
        scalar_t timeLeft = timeLeftInPhase(legidx);
        scalar_t eventTime = currentTime + timeLeft;
        vector_t optimizedState =
            LinearInterpolation::interpolate(eventTime, solution.timeTrajectory_, solution.stateTrajectory_);
        auto &quadcom = *comModel_;
        auto &kin = *kinematicsModel_;
        auto basePoseOcs2 = switched_model::getBasePose(optimizedState);
        auto jointAnglesOcs2 = switched_model::getJointPositions(optimizedState);
        vector3_t futureFootPosition = kin.footPositionInOriginFrame(legidx, basePoseOcs2, jointAnglesOcs2);
        vector3_t currentFootPosition = currentFootPositions[legidx];

        // Update desired footholds
        matrix3_t R_base_world = getRotationMatrixBaseWorld(rbdState);
        vector_t footholdInBase = R_base_world * (futureFootPosition - currentFootPosition);

        out.segment<2>(2 * legidx) = footholdInBase.head<2>();
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
            LinearInterpolation::interpolate(currentTime + timeLeft, solution.timeTrajectory_, solution.stateTrajectory_);

        auto *quadcomPtr = dynamic_cast<anymal::QuadrupedCom *>(comModel_.get());
        auto &quadcom = *quadcomPtr;
        auto &kin = *kinematicsModel_;

        auto basePoseOcs2 = switched_model::getBasePose(optimizedState);
        auto jointAnglesOcs2 = switched_model::getJointPositions(optimizedState);

        auto qPinocchio = quadcom.getPinnochioConfiguration(basePoseOcs2, jointAnglesOcs2);
        auto optimizedJointAngles = qPinocchio.tail<12>();

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
    auto *quadcomPtr = dynamic_cast<anymal::QuadrupedCom *>(comModel_.get());
    auto &quadcom = *quadcomPtr;
    auto &kin = *kinematicsModel_;

    auto basePoseOcs2 = switched_model::getBasePose(optimizedState);
    auto jointAnglesOcs2 = switched_model::getJointPositions(optimizedState);

    auto qPinocchio = quadcom.getPinnochioConfiguration(basePoseOcs2, jointAnglesOcs2);
    auto jointAngles = qPinocchio.tail<12>();
    return jointAngles - defaultJointAngles_;
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
    vector3_t basePos, baseLinearVelocity, baseAngularVelocity, baseLinearAcceleration,
        baseAngularAcceleration;
    vector_t baseOrientation;
    computeBaseKinematicsAndDynamics(currentTime, dt, basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity,
                                     baseLinearAcceleration, baseAngularAcceleration);
    const auto &rbdState = stateSubscriberPtr_->getLatestRbdState();
    vector3_t basePosCurrent = rbdState.segment<3>(3);
    vector3_t basePosDesired = basePos;
    matrix3_t R_base_world = getRotationMatrixBaseWorld(rbdState);
    vector3_t out = R_base_world * (basePosDesired - basePosCurrent);
    return out;
}

vector_t DtcController::getOrientationDiffObservation(scalar_t currentTime, scalar_t dt) {
    vector3_t basePos, baseLinearVelocity, baseAngularVelocity, baseLinearAcceleration,
        baseAngularAcceleration;
    vector_t baseOrientation;
    computeBaseKinematicsAndDynamics(currentTime, dt, basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity,
                                     baseLinearAcceleration, baseAngularAcceleration);
    const auto &rbdState = stateSubscriberPtr_->getLatestRbdState();

    vector3_t eulerAnglesZyxCurrent = getOcs2ZyxEulerAngles(rbdState);
    quaternion_t quatCurrent = this->getQuaternionFromEulerAnglesZyx(eulerAnglesZyxCurrent);

    const scalar_t xdes = baseOrientation(0);
    const scalar_t ydes = baseOrientation(1);
    const scalar_t zdes = baseOrientation(2);
    const scalar_t wdes = baseOrientation(3);
    quaternion_t quatDesired = quaternion_t(wdes, xdes, ydes, zdes);

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
    vector3_t basePos, baseLinearVelocity, baseAngularVelocity, baseLinearAcceleration,
        baseAngularAcceleration;
    vector_t baseOrientation;
    computeBaseKinematicsAndDynamics(currentTime, dt, basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity,
                                     baseLinearAcceleration, baseAngularAcceleration);
    const auto &rbdState = stateSubscriberPtr_->getLatestRbdState();
    matrix3_t R_base_world = getRotationMatrixBaseWorld(rbdState);
    vector3_t baseLinVelDesiredWorld = baseLinearVelocity;
    vector3_t baseLinVelDesiredBase = R_base_world * baseLinVelDesiredWorld;
    return baseLinVelDesiredBase;
}

vector_t DtcController::getDesiredBaseAngVelObservation(scalar_t currentTime, scalar_t dt) {
    vector3_t basePos, baseLinearVelocity, baseAngularVelocity, baseLinearAcceleration,
        baseAngularAcceleration;
    vector_t baseOrientation;
    computeBaseKinematicsAndDynamics(currentTime, dt, basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity,
                                     baseLinearAcceleration, baseAngularAcceleration);
    const auto &rbdState = stateSubscriberPtr_->getLatestRbdState();
    matrix3_t R_base_world = getRotationMatrixBaseWorld(rbdState);
    vector3_t baseAngVelDesiredWorld = baseAngularVelocity;
    vector3_t baseAngVelDesiredBase = R_base_world * baseAngVelDesiredWorld;
    return baseAngVelDesiredBase;
}

vector_t DtcController::getDesiredBaseLinAccObservation(scalar_t currentTime, scalar_t dt) {
    vector3_t basePos, baseLinearVelocity, baseAngularVelocity, baseLinearAcceleration,
        baseAngularAcceleration;
    vector_t baseOrientation;
    computeBaseKinematicsAndDynamics(currentTime, dt, basePos, baseOrientation, baseLinearVelocity, baseAngularVelocity,
                                     baseLinearAcceleration, baseAngularAcceleration);
    const auto &rbdState = stateSubscriberPtr_->getLatestRbdState();
    matrix3_t R_base_world = getRotationMatrixBaseWorld(rbdState);
    vector3_t baseLinAccDesiredWorld = baseLinearAcceleration;
    vector3_t baseLinAccDesiredBase = R_base_world * baseLinAccDesiredWorld;
    return baseLinAccDesiredBase;
}

vector_t DtcController::getDesiredBaseAngAccObservation(scalar_t currentTime, scalar_t dt) {
    vector3_t basePos, baseLinearVelocity, baseAngularVelocity, baseLinearAcceleration,
        baseAngularAcceleration;
    vector_t baseOrientation;
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

    matrix3_t R_base_world = getRotationMatrixBaseWorld(stateSubscriberPtr_->getLatestRbdState());

    vector_t lf_pos = -R_base_world * (desiredFootPositions[0] - currentFootPositions[0]);
    vector_t rf_pos = -R_base_world * (desiredFootPositions[1] - currentFootPositions[1]);
    vector_t lh_pos = -R_base_world * (desiredFootPositions[2] - currentFootPositions[2]);
    vector_t rh_pos = -R_base_world * (desiredFootPositions[3] - currentFootPositions[3]);

    vector_t out(4 * 3);
    out << lf_pos, lh_pos, rf_pos, rh_pos;
    return out;
}
vector_t DtcController::getDesiredFootVelocitiesObservation(scalar_t currentTime, scalar_t dt) {
    auto desiredFootVelocities = getDesiredFeetVelocities(currentTime, dt);
    auto currentFootVelocities = getCurrentFeetVelocities(currentTime, dt);

    matrix3_t R_base_world = getRotationMatrixBaseWorld(stateSubscriberPtr_->getLatestRbdState());

    vector_t lf_vel = -R_base_world * (desiredFootVelocities[0] - currentFootVelocities[0]);
    vector_t rf_vel = -R_base_world * (desiredFootVelocities[1] - currentFootVelocities[1]);
    vector_t lh_vel = -R_base_world * (desiredFootVelocities[2] - currentFootVelocities[2]);
    vector_t rh_vel = -R_base_world * (desiredFootVelocities[3] - currentFootVelocities[3]);

    vector_t out(4 * 3);
    out << lf_vel, lh_vel, rf_vel, rh_vel;
    return out;
}

vector_t DtcController::getHeightSamplesObservation(scalar_t currentTime, scalar_t dt) {
    auto &solution = mrt_.getPolicy();
    auto currentFootPositions = getCurrentFeetPositions(currentTime, dt);
    auto timeLeftInPhase = getTimeLeftInPhase(currentTime, dt);
    vector_t out(4*10);
    for(int legidx = 0; legidx < 4; ++legidx) {
        scalar_t timeLeft = timeLeftInPhase(legidx);
        scalar_t eventTime = currentTime + timeLeft;
        vector_t optimizedState =
            LinearInterpolation::interpolate(eventTime, solution.timeTrajectory_, solution.stateTrajectory_);
        auto &quadcom = *comModel_;
        auto &kin = *kinematicsModel_;
        auto basePoseOcs2 = switched_model::getBasePose(optimizedState);
        auto jointAnglesOcs2 = switched_model::getJointPositions(optimizedState);
        vector3_t futureFootPosition = kin.footPositionInOriginFrame(legidx, basePoseOcs2, jointAnglesOcs2);
        vector3_t currentFootPosition = currentFootPositions[legidx];

        vector3_t diff = futureFootPosition - currentFootPosition;
        scalar_t dalpha = 1.0 / (10 - 1);
        for(int i = 0; i < 10; ++i) {
            scalar_t alpha = dalpha * i;
            vector3_t pos = currentFootPosition + diff * alpha;
            scalar_t x = pos(0);
            scalar_t y = pos(1);
            scalar_t height = gridmap_->atPosition(x,y);
            scalar_t height_diff = height - currentFootPosition(2);
            out(legidx * 10 + i) = height_diff;
        }
    }
    return out;
}


void DtcController::visualize() {
    mrt_.spinMRT();
    mrt_.updatePolicy();
    visualizer_->update(generateSystemObservation(), mrt_.getPolicy(), mrt_.getCommand());
}

void DtcController::changeController(const std::string &controllerType, scalar_t currentTime) {
    DTC_PRINT("Changing controller");
    resetMpc();
    DTC_PRINT("Controller changed");
    if (!blind_) {
        gridmap_->waitTillInitialized();
    }
}

bool DtcController::isSupported(const std::string &controllerType) {
    return controllerType == "DTC";
}

ocs2::SystemObservation DtcController::generateSystemObservation() {
    const tbai::vector_t &rbdState = stateSubscriberPtr_->getLatestRbdState();

    // Set observation time
    ocs2::SystemObservation observation;
    observation.time = stateSubscriberPtr_->getLatestRbdStamp().toSec() - initTime_;

    // Set mode
    observation.mode = switched_model::stanceLeg2ModeNumber(stateSubscriberPtr_->getContactFlags());

    // Set state
    observation.state = rbdState.head<3 + 3 + 3 + 3 + 12>();

    // Swap LH and RF
    std::swap(observation.state(3 + 3 + 3 + 3 + 3 + 0), observation.state(3 + 3 + 3 + 3 + 3 + 3));
    std::swap(observation.state(3 + 3 + 3 + 3 + 3 + 1), observation.state(3 + 3 + 3 + 3 + 3 + 4));
    std::swap(observation.state(3 + 3 + 3 + 3 + 3 + 2), observation.state(3 + 3 + 3 + 3 + 3 + 5));

    // Set input
    observation.input.setZero(24);
    observation.input.tail<12>() = rbdState.tail<12>();

    // Swap LH and RF
    std::swap(observation.input(12 + 3), observation.input(12 + 6));
    std::swap(observation.input(12 + 4), observation.input(12 + 7));
    std::swap(observation.input(12 + 5), observation.input(12 + 8));

    return observation;
}

void DtcController::resetMpc() {
    DTC_PRINT("Waiting for state subscriber to initialize...");

    // Wait to receive observation
    stateSubscriberPtr_->waitTillInitialized();

    DTC_PRINT("State subscriber initialized");

    // Prepare initial observation for MPC
    ocs2::SystemObservation mpcObservation = generateSystemObservation();

    DTC_PRINT("Initial observation generated");

    // Prepare target trajectory
    ocs2::TargetTrajectories initTargetTrajectories({0.0}, {mpcObservation.state}, {mpcObservation.input});

    DTC_PRINT("Resetting MPC...");
    mrt_.resetMpcNode(initTargetTrajectories);

    while (!mrt_.initialPolicyReceived() && ros::ok()) {
        ROS_INFO("Waiting for initial policy...");
        DTC_PRINT("Waiting for initial policy...");
        ros::spinOnce();
        mrt_.spinMRT();
        mrt_.setCurrentObservation(generateSystemObservation());
        ros::Duration(0.1).sleep();
    }

    DTC_PRINT("Initial policy received");

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
