#ifndef TBAI_GAZEBO_INCLUDE_TBAI_GAZEBO_STATEPUBLISHER_HPP_
#define TBAI_GAZEBO_INCLUDE_TBAI_GAZEBO_STATEPUBLISHER_HPP_

#include <vector>

#include <Eigen/Dense>
#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ros/ros.h>

namespace gazebo {
class StatePublisher : public ModelPlugin {
   public:
    void Load(physics::ModelPtr _parent, sdf::ElementPtr sdf);
    void OnUpdate();

   private:
    /** Convert rotation matrix to ZYX euler angles - output {roll, pitch, yaw}*/
    Eigen::Vector3d eulerXYZFromRotationMatrix(const Eigen::Matrix3d &R, double lastYaw);

    event::ConnectionPtr updateConnection_;

    /** RbdState message publisher */
    ros::Publisher statePublisher_;

    /** Robot gazebo model */
    physics::ModelPtr robot_;

    /** Base link */
    physics::LinkPtr baseLink_;

    /** Joints */
    std::vector<physics::JointPtr> joints_;

    /** State publish rate */
    double rate_;
    double period_;

    /** Last time a state message was published */
    common::Time lastPublishTime_;

    /** Last yaw euler angle */
    double lastYaw_;
};

}  // namespace gazebo

#endif  // TBAI_GAZEBO_INCLUDE_TBAI_GAZEBO_STATEPUBLISHER_HPP_
