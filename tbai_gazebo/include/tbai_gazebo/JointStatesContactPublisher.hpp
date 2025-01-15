#pragma once

#include <vector>

#include <Eigen/Dense>
#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <tbai_core/Types.hpp>

namespace gazebo {
class JointStatesContactPublisher : public ModelPlugin {
   public:
    void Load(physics::ModelPtr _parent, sdf::ElementPtr sdf);
    void OnUpdate();

   private:
    event::ConnectionPtr updateConnection_;

    /** JointStates message publisher */
    ros::Publisher jointStatePublisher_;

    /** contacts flags publisher*/
    ros::Publisher contactFlagPublisher_;

    /** Robot gazebo model */
    physics::ModelPtr robot_;

    /** Base link */
    physics::LinkPtr baseLinkPtr_;

    std::vector<physics::JointPtr> joints_;

    /** State publish rate */
    double rate_;
    double period_;

    bool firstUpdate_ = true;

    double lastYaw_ = 0.0;

    std::array<bool, 4> contactFlags_;
    std::array<ros::Subscriber, 4> contactSubscribers_;

    // last yaw angle
    std::vector<tbai::scalar_t> lastJointAngles_;
    common::Time lastSimTime_;
};

}  // namespace gazebo
