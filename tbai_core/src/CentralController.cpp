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
        // Start of epoch
        initTime_ = ros::Time::now();

        // Main loop rate
        loopRate_ = ros::Rate(activeController_->getRate());
    }

    while (ros::ok()) {
        ros::spinOnce();
        step(getCurrentTime());
        visualize();
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
void CentralController::step(scalar_t currentTime) {
    auto commandMessage = activeController_->getCommandMessage(currentTime);
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
            ROS_INFO_STREAM("Changed controller to " << controllerType);
            return;
        }
    }
}

}  // namespace core
}  // namespace tbai
