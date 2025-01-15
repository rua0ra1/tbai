// clang-format off
#include <pinocchio/fwd.hpp>
// clang-format on

#include "tbai_gazebo/JointStatesContactPublisher.hpp"

#include <functional>
#include <string>

#include <Eigen/Geometry>
#include <tbai_core/Rotations.hpp>
#include <tbai_core/config/YamlConfig.hpp>
#include <tbai_msgs/JointStates.h>
#include <tbai_msgs/Contacts.h>

namespace gazebo {

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void JointStatesContactPublisher::Load(physics::ModelPtr robot, sdf::ElementPtr sdf) {
    ROS_INFO_STREAM("[StatePublisher] Loading StatePublisher plugin");
    // set Gazebo callback function
    updateConnection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&JointStatesContactPublisher::OnUpdate, this));

    this->robot_ = robot;
    auto config = tbai::core::YamlConfig::fromRosParam("/tbai_config_path");

    ros::NodeHandle nh;
    //Joint state publisher
    auto stateTopic = config.get<std::string>("JointStatesTopic");
    jointStatePublisher_ = nh.advertise<tbai_msgs::JointStates>(stateTopic, 2);
    // contact flag publisher
    auto contactTopic = config.get<std::string>("ContactFlagTopic");
    contactFlagPublisher_ = nh.advertise<tbai_msgs::Contacts>(contactTopic, 2);

    auto base = config.get<std::string>("base_name");
    baseLinkPtr_ = robot->GetChildLink(base);

    // get joints; ignore 'universe' and 'root_joint'
    auto jointNames = config.get<std::vector<std::string>>("joint_names");
    for (int i = 0; i < jointNames.size(); ++i) {
        joints_.push_back(robot->GetJoint(jointNames[i]));
    }

    // initialize last publish time
    lastSimTime_ = robot->GetWorld()->SimTime();

    rate_ = config.get<double>("stateUpdateFreq");
    period_ = 1.0 / rate_;

    // get contact topics
    auto contactTopics = config.get<std::vector<std::string>>("contact_topics");
    for (int i = 0; i < contactTopics.size(); ++i) {
        contactFlags_[i] = false;
        auto callback = [this, i](const std_msgs::Bool::ConstPtr &msg) { contactFlags_[i] = msg->data; };
        contactSubscribers_[i] = nh.subscribe<std_msgs::Bool>(contactTopics[i], 1, callback);
    }

    ROS_INFO_STREAM("[JointStateContactPublisher] Loaded JointStateContactPublisher plugin");
}  // namespace gazebo

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void JointStatesContactPublisher::OnUpdate() {
    // Get current time
    const common::Time currentTime = robot_->GetWorld()->SimTime();
    const double dt = (currentTime - lastSimTime_).Double();

    // Check if update is needed
    if (dt < period_) {
        return;
    }

    // Joint angles
    std::vector<double> jointAngles(joints_.size());
    for (int i = 0; i < joints_.size(); ++i) {
        jointAngles[i] = joints_[i]->Position(0);
    }

    // Joint velocities
    if (lastJointAngles_.size() != joints_.size()) {
        lastJointAngles_.resize(joints_.size());
        for (size_t i = 0; i < joints_.size(); ++i) {
            lastJointAngles_[i] = jointAngles[i];
        }
    }

    // Get joint velocities
    std::vector<double> jointVelocities(joints_.size());
    for (int i = 0; i < joints_.size(); ++i) {
        jointVelocities[i] = (jointAngles[i] - lastJointAngles_[i]) / dt;
        lastJointAngles_[i] = jointAngles[i];
    }

    // update joint states and velocity into an joint message
    tbai_msgs::JointStates joint_state_msg; 

    // Joint positions and velocities
    for (int i = 0; i < jointAngles.size(); ++i) {
        joint_state_msg.joints_position[i] = jointAngles[i];
        joint_state_msg.joints_velocity[i] = jointVelocities[i];
    }
    // Observation time
    joint_state_msg.stamp = ros::Time::now();
    // Publish message
    jointStatePublisher_.publish(joint_state_msg);

    
    /* update the contact flags */
    tbai_msgs::Contacts contact_msg; 
    // Contact flags
    std::copy(contactFlags_.begin(), contactFlags_.end(), contact_msg.contact_flags.begin());
    // Observation time
    contact_msg.stamp = ros::Time::now();

    // Publish message
    contactFlagPublisher_.publish(contact_msg);

    lastSimTime_ = currentTime;
}

GZ_REGISTER_MODEL_PLUGIN(JointStatesContactPublisher);

}  // namespace gazebo
