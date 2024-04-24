
#include "tbai_mpc/MpcController.hpp"

#include <string>
#include <vector>

#include <ocs2_switched_model_interface/core/MotionPhaseDefinition.h>

namespace tbai {
namespace mpc {

MpcController::MpcController(const std::shared_ptr<tbai::core::StateSubscriber> &stateSubscriberPtr)
    : stateSubscriberPtr_(stateSubscriberPtr), mrt_("anymal") {
    initTime_ = ros::Time::now().toSec();

    const std::string robotName = "anymal";
    ros::NodeHandle nh;

    // Load default joint state
    std::string targetCommandConfig =
        "/home/kuba/fun/ocs2_project/src/ocs2_fun/ocs2_anymal_robot/config/targetCommand.info";

    // Description name
    std::string descriptionName = "robot_description";

    // URDF
    std::string urdfString;
    nh.getParam(descriptionName, urdfString);

    // Task settings
    std::string taskSettingsFile = "/home/kuba/fun/ocs2_project/src/ocs2_fun/ocs2_anymal_robot/config/task.info";

    // Frame declarations
    std::string frameDeclarationFile =
        "/home/kuba/fun/ocs2_project/src/ocs2_fun/ocs2_anymal_robot/config/frame_declarations.info";

    // Controller config
    std::string controllerConfigFile =
        "/home/kuba/fun/ocs2_project/src/ocs2_fun/ocs2_anymal_robot/config/controllers.info";

    const std::string stateTopic = "/" + descriptionName + "/" + robotName + "/state";

    quadrupedInterfacePtr_ =
        anymal::getAnymalInterface(urdfString, switched_model::loadQuadrupedSettings(taskSettingsFile),
                                   anymal::frameDeclarationFromFile(frameDeclarationFile));

    visualizerPtr_ = std::make_shared<switched_model::QuadrupedVisualizer>(quadrupedInterfacePtr_->getKinematicModel(),
                                                                           quadrupedInterfacePtr_->getJointNames(),
                                                                           quadrupedInterfacePtr_->getBaseName(), nh);

    wbcPtr_ = std::make_unique<switched_model::SqpWbc>(
        controllerConfigFile, urdfString, quadrupedInterfacePtr_->getComModel(),
        quadrupedInterfacePtr_->getKinematicModel(), quadrupedInterfacePtr_->getJointNames());

    mrt_.launchNodes(nh);
    tNow_ = 0.0;
}

tbai_msgs::JointCommandArray MpcController::getCommandMessage(scalar_t currentTime, scalar_t dt) {
    // std::vector<std::string> jointNames = {"LF_HAA", "LF_HFE", "LF_KFE", "LH_HAA", "LH_HFE", "LH_KFE",
    //                                        "RF_HAA", "RF_HFE", "RF_KFE", "RH_HAA", "RH_HFE", "RH_KFE"};

    // std::vector<scalar_t> jointAngles = {0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8};

    // tbai_msgs::JointCommandArray jointCommandArray;
    // jointCommandArray.joint_commands.resize(12);
    // for (int i = 0; i < 12; ++i) {
    //     auto &command = jointCommandArray.joint_commands[i];
    //     command.joint_name = jointNames[i];
    //     command.desired_position = jointAngles[i];
    //     command.desired_velocity = 0.0;
    //     command.torque_ff = 0.0;
    //     command.kp = 400;
    //     command.kd = 10;
    // }

    // std::cout << "MPC controller" << std::endl;
    // auto obs = generateSystemObservation();
    // std::cout << obs << std::endl;

    // return jointCommandArray;

    mrt_.spinMRT();
    mrt_.updatePolicy();

    tNow_ += dt;

    auto observation = generateSystemObservation();

    ocs2::vector_t desiredState;
    ocs2::vector_t desiredInput;
    size_t desiredMode;
    mrt_.evaluatePolicy(tNow_, observation.state, desiredState, desiredInput, desiredMode);

    constexpr ocs2::scalar_t time_eps = 1e-4;
    ocs2::vector_t dummyState;
    ocs2::vector_t dummyInput;
    size_t dummyMode;
    mrt_.evaluatePolicy(tNow_ + time_eps, observation.state, dummyState, dummyInput, dummyMode);

    ocs2::vector_t joint_accelerations = (dummyInput.tail<12>() - desiredInput.tail<12>()) / time_eps;

    auto commandMessage = wbcPtr_->getCommandMessage(tNow_, observation.state, observation.input, observation.mode,
                                                     desiredState, desiredInput, desiredMode, joint_accelerations);

    timeSinceLastMpcUpdate_ += dt;
    if (timeSinceLastMpcUpdate_ >= 1.0 / mpcRate_) {
        setObservation();
    }

    return commandMessage;
}

void MpcController::visualize() {}

void MpcController::changeController(const std::string &controllerType, scalar_t currentTime) {
    if (!mrt_initialized_ || currentTime + 0.1 > mrt_.getPolicy().timeTrajectory_.back()) {
        resetMpc();
        mrt_initialized_ = true;
    }
    tNow_ = currentTime;
}

bool MpcController::isSupported(const std::string &controllerType) {
    return controllerType == "SQP_WBC";
}

void MpcController::resetMpc() {
    // Generate initial observation
    stateSubscriberPtr_->waitTillInitialized();
    auto initialObservation = generateSystemObservation();
    const ocs2::TargetTrajectories initTargetTrajectories({0.0}, {initialObservation.state},
                                                          {initialObservation.input});
    mrt_.resetMpcNode(initTargetTrajectories);

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

void MpcController::setObservation() {
    mrt_.setCurrentObservation(generateSystemObservation());
    timeSinceLastMpcUpdate_ = 0.0;
}

ocs2::SystemObservation MpcController::generateSystemObservation() const {
    const tbai::vector_t &rbdState = stateSubscriberPtr_->getLatestRbdState();

    // Set observation time
    ocs2::SystemObservation observation;
    observation.time = ros::Time::now().toSec() - initTime_;  // TODO: Replace with actual observation stamp

    // Set mode
    std::array<bool, 4> contacts = {false, false, false, false};
    observation.mode = switched_model::stanceLeg2ModeNumber(contacts);
    observation.mode = 14;

    // Set state
    observation.state = rbdState.head<3 + 3 + 3 + 3 + 12>();

    // Swap LH and RF
    std::swap(observation.state(3 + 3 + 3 + 3 + 3 + 0), observation.state(3 + 3 + 3 + 3 + 3 + 3));
    std::swap(observation.state(3 + 3 + 3 + 3 + 3 + 1), observation.state(3 + 3 + 3 + 3 + 3 + 4));
    std::swap(observation.state(3 + 3 + 3 + 3 + 3 + 2), observation.state(3 + 3 + 3 + 3 + 3 + 5));

    // Set input
    observation.input.setZero(24);
    observation.input.tail<12>() = rbdState.tail<12>();

    std::swap(observation.input(12 + 3), observation.input(12 + 6));
    std::swap(observation.input(12 + 4), observation.input(12 + 7));
    std::swap(observation.input(12 + 5), observation.input(12 + 8));

    return observation;
}

}  // namespace mpc
}  // namespace tbai
