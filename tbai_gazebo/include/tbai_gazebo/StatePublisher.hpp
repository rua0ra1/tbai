#ifndef TBAI_GAZEBO_INCLUDE_TBAI_GAZEBO_STATEPUBLISHER_HPP_
#define TBAI_GAZEBO_INCLUDE_TBAI_GAZEBO_STATEPUBLISHER_HPP_

#include <vector>

#include <Eigen/Dense>
#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ros/ros.h>
#include <tbai_core/Types.hpp>

namespace gazebo {
class StatePublisher : public ModelPlugin {
   public:
    void Load(physics::ModelPtr _parent, sdf::ElementPtr sdf);
    void OnUpdate();

   private:
    /* Convert a rotation matrix to its angle-axis representation */
    tbai::vector3_t mat2aa(const tbai::matrix3_t &R);

    event::ConnectionPtr updateConnection_;

    /** RbdState message publisher */
    ros::Publisher statePublisher_;

    /** Robot gazebo model */
    physics::ModelPtr robot_;

    /** Base link */
    physics::LinkPtr baseLinkPtr_;

    std::vector<physics::JointPtr> joints_;

    /** State publish rate */
    double rate_;
    double period_;

    bool firstUpdate_ = true;

    // last yaw angle
    std::vector<tbai::scalar_t> lastJointAngles_;
    tbai::matrix3_t lastBaseOrientationMat_;
    tbai::vector3_t lastBasePosition_;
    common::Time lastSimTime_;
};

}  // namespace gazebo

#endif  // TBAI_GAZEBO_INCLUDE_TBAI_GAZEBO_STATEPUBLISHER_HPP_
