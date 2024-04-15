#include "tbai_static/StaticController.hpp"

#include <Eigen/Core>
#include <geometry_msgs/TransformStamped.h>
#include <kdl_parser/kdl_parser.hpp>
#include <ros/package.h>
#include <tbai_config/YamlConfig.hpp>
#include <urdf/model.h>

namespace tbai {
namespace core {

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
StaticController::StaticController(const std::string &configRosParam,
                                   std::shared_ptr<StateSubscriber> stateSubscriberPtr, scalar_t initialTime)
    : stateSubscriberPtr_(stateSubscriberPtr), lastTime_(initialTime), alpha_(-1.0), currentControllerType_("SIT") {
    loadSettings(configRosParam);

    // Initialize robot state publisher
    const std::string urdfFile = ros::package::getPath("ocs2_robotic_assets") + "/resources/anymal_d/urdf/anymal.urdf";
    urdf::Model urdfModel;
    KDL::Tree kdlTree;
    kdl_parser::treeFromFile(urdfFile, kdlTree);
    robotStatePublisherPtr_.reset(new robot_state_publisher::RobotStatePublisher(kdlTree));
    robotStatePublisherPtr_->publishFixedTransforms();
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
tbai_msgs::JointCommandArray StaticController::getCommandMessage(scalar_t currentTime) {}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void StaticController::visualize() {
    ros::Time currentTime = ros::Time::now();
    const vector_t &currentState = stateSubscriberPtr_->getLatestRbdState();
    publishOdomBaseTransforms(currentState, currentTime);
    publishJointAngles(currentState, currentTime);
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void StaticController::changeController(const std::string &controllerType, scalar_t currentTime) {
    currentControllerType_ = controllerType;
    alpha_ = 0.0;
    interpFrom_ = stateSubscriberPtr_->getLatestRbdState().segment<12>(6);
    if (currentControllerType_ == "STAND") {
        interpTo_ = standJointAngles_;
    } else if (currentControllerType_ == "SIT") {
        interpTo_ = sitJointAngles_;
    } else {
        throw std::runtime_error("Unsupported controller type: " + currentControllerType_);
    }
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
bool StaticController::isSupported(const std::string &controllerType) {
    if (controllerType == "STAND" || controllerType == "SIT") {
        return true;
    }
    return false;
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
scalar_t StaticController::getRate() const {
    return rate_;
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void StaticController::loadSettings(const std::string &configRosParam) {
    auto config = tbai::config::YamlConfig::fromRosParam(configRosParam, '/');
    kp_ = config.get<scalar_t>("static_controller/kp");
    kd_ = config.get<scalar_t>("static_controller/kd");
    rate_ = config.get<scalar_t>("static_controller/rate");

    standJointAngles_ = config.get<vector_t>("static_controller/stand_controller/joint_angles");
    sitJointAngles_ = config.get<vector_t>("static_controller/sit_controller/joint_angles");
    jointNames_ = config.get<std::vector<std::string>>("joint_names");
    interpolationTime_ = config.get<scalar_t>("static_controller/interpolation_time");
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void StaticController::publishOdomBaseTransforms(const vector_t &currentState, const ros::Time &currentTime) {
    geometry_msgs::TransformStamped odomBaseTransform;

    // Header
    odomBaseTransform.header.stamp = currentTime;
    odomBaseTransform.header.frame_id = "odom";
    odomBaseTransform.child_frame_id = "base";

    // Position
    odomBaseTransform.transform.translation.x = currentState(3);
    odomBaseTransform.transform.translation.y = currentState(4);
    odomBaseTransform.transform.translation.z = currentState(5);

    // Orientation
    Eigen::Quaternion<scalar_t> quaternion =
        (Eigen::AngleAxis<scalar_t>(currentState(0), Eigen::Matrix<scalar_t, 3, 1>::UnitZ()) *
         Eigen::AngleAxis<scalar_t>(currentState(1), Eigen::Matrix<scalar_t, 3, 1>::UnitY()) *
         Eigen::AngleAxis<scalar_t>(currentState(2), Eigen::Matrix<scalar_t, 3, 1>::UnitX()));
    odomBaseTransform.transform.rotation.x = quaternion.x();
    odomBaseTransform.transform.rotation.y = quaternion.y();
    odomBaseTransform.transform.rotation.z = quaternion.z();
    odomBaseTransform.transform.rotation.w = quaternion.w();

    // Publish
    tfBroadcaster_.sendTransform(odomBaseTransform);
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void StaticController::publishJointAngles(const vector_t &currentState, const ros::Time &currentTime) {
    std::map<std::string, scalar_t> jointPositionMap;
    for (size_t i = 0; i < jointNames_.size(); ++i) {
        jointPositionMap[jointNames_[i]] = currentState(i + 6);
    }
    robotStatePublisherPtr_->publishTransforms(jointPositionMap, currentTime);
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
tbai_msgs::JointCommandArray StaticController::getInterpCommandMessage(scalar_t dt) {
    // Update alpha
    alpha_ = std::min(alpha_ + dt / interpolationTime_, static_cast<scalar_t>(1.0));

    // Compute new joint angles
    auto jointAngles = (1.0 - alpha_) * interpFrom_ + alpha_ * interpTo_;

    // Finish interpolation
    if (alpha_ == 1.0) {
        alpha_ = -1.0;
    }

    return packCommandMessage(jointAngles);
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
tbai_msgs::JointCommandArray StaticController::getStandCommandMessage() {
    return packCommandMessage(standJointAngles_);
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
tbai_msgs::JointCommandArray StaticController::getSitCommandMessage() {
    return packCommandMessage(sitJointAngles_);
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
tbai_msgs::JointCommandArray StaticController::packCommandMessage(const vector_t &jointAngles) {
    tbai_msgs::JointCommandArray commandArray;
    commandArray.joint_commands.resize(jointAngles.size());
    for (size_t i = 0; i < jointAngles.size(); ++i) {
        tbai_msgs::JointCommand command;
        command.joint_name = jointNames_[i];
        command.desired_position = jointAngles[i];
        command.desired_velocity = 0.0;
        command.kp = kp_;
        command.kd = kd_;
        command.torque_ff = 0.0;
        commandArray.joint_commands[i] = command;
    }
    return commandArray;
}

}  // namespace core
}  // namespace tbai
