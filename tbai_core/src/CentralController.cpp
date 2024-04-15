#include "tbai_core/CentralController.hpp"

#include <ros/ros.h>
#include <tbai_msgs/JointCommandArray.h>

namespace tbai {
namespace core {

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
CentralController::CentralController(ros::NodeHandle &nh, const std::string &stateTopic,
                                     const std::string &commandTopic, const std::string &changeControllerTopic)
    : loopRate_(1), activeController_(nullptr) {
    stateSubscriberPtr_ = std::make_shared<StateSubscriber>(nh, stateTopic);
    commandPublisher_ = nh.advertise<tbai_msgs::JointCommandArray>(commandTopic, 1);
    changeControllerSubscriber_ =
        nh.subscribe(changeControllerTopic, 1, &CentralController::changeControllerCallback, this);
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void CentralController::start() {
    // Make sure there is an active controller
    if (activeController_ == nullptr) {
        ROS_ERROR("No active controller found");
        return;
    }

    // Wait for initial state message
    stateSubscriberPtr_->waitTillInitialized();

    if (ros::ok()) {
        // Main loop rate
        loopRate_ = ros::Rate(activeController_->getRate());

        // Start of epoch
        initTime_ = ros::Time::now();
    }

    scalar_t lastTime = 0.0;
    while (ros::ok()) {
        // Spin once to allow ROS run callbacks
        ros::spinOnce();

        // Compute current time and time since last call
        scalar_t currentTime = getCurrentTime();
        scalar_t dt = currentTime - lastTime;

        // Step controller
        step(currentTime, dt);

        // Allow controller to visualize stuff
        visualize();

        lastTime = currentTime;
        loopRate_.sleep();
    }
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void CentralController::addController(std::unique_ptr<Controller> controller, bool makeActive) {
    controllers_.push_back(std::move(controller));
    if (makeActive || activeController_ == nullptr) {
        activeController_ = controllers_.back().get();
    }
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void CentralController::step(scalar_t currentTime, scalar_t dt) {
    auto commandMessage = activeController_->getCommandMessage(currentTime, dt);
    commandPublisher_.publish(commandMessage);
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
std::shared_ptr<StateSubscriber> CentralController::getStateSubscriberPtr() {
    return stateSubscriberPtr_;
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void CentralController::changeControllerCallback(const std_msgs::String::ConstPtr &msg) {
    const std::string controllerType = msg->data;
    for (auto &controller : controllers_) {
        if (controller->isSupported(controllerType)) {
            activeController_ = controller.get();
            activeController_->changeController(controllerType, getCurrentTime());
            loopRate_ = ros::Rate(activeController_->getRate());
            ROS_INFO_STREAM("[CentralController] Controller changed to " << controllerType);
            return;
        }
    }
    ROS_WARN_STREAM("[CentralController] Controller " << controllerType << " not supported");
}

}  // namespace core
}  // namespace tbai
