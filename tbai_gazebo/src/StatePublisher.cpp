#include "tbai_gazebo/StatePublisher.hpp"

#include <functional>
#include <string>

#include <Eigen/Geometry>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <tbai_msgs/RbdState.h>

namespace gazebo {

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void StateEstimator::Load(physics::ModelPtr robot, sdf::ElementPtr sdf) {
    // set Gazebo callback function
    updateConnection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&StateEstimator::OnUpdate, this));

    this->robot_ = robot;

    ros::NodeHandle nh;
    statePublisher_ = nh.advertise<tbai_msgs::RbdState>("anymal/state", 2);

    std::string base = "base";
    baseLink_ = robot->GetChildLink(base);

    // get joints; ignore 'universe' and 'root_joint'
    std::vector<std::string> jointNames = {"LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE",
                                           "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE"};
    for (int i = 0; i < jointNames.size(); ++i) {
        joints_.push_back(robot->GetJoint(jointNames[i]));
    }

    // initialize last publish time
    lastPublishTime_ = robot->GetWorld()->SimTime();

    lastYaw_ = 0.0;

    rate_ = 400.0;
    period_ = 1.0 / rate_;
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void StateEstimator::OnUpdate() {
    // Get current time
    const common::Time currentTime = robot_->GetWorld()->SimTime();
    const double dt = (currentTime - lastPublishTime_).Double();

    // Check if update is needed
    if (dt < period_) {
        return;
    }

    // Base orientation - ZYX Euler angles
    const ignition::math::Pose3d &basePose = baseLink_->WorldPose();
    Eigen::Quaternion<double> baseQuaternion(basePose.Rot().W(), basePose.Rot().X(), basePose.Rot().Y(),
                                             basePose.Rot().Z());
    const Eigen::Matrix3d R_world_base = baseQuaternion.toRotationMatrix();
    const Eigen::Vector3d rpy = eulerXYZFromRotationMatrix(R_world_base, lastYaw_);
    lastYaw_ = rpy[2];  // rpy = {roll, pitch, yaw}

    // Base position - XYZ world coordinates
    const Eigen::Vector3d basePosition(basePose.Pos().X(), basePose.Pos().Y(), basePose.Pos().Z());

    // Joint angles
    std::vector<double> jointAngles(joints_.size());
    for (int i = 0; i < joints_.size(); ++i) {
        jointAngles[i] = joints_[i]->Position(0);
    }

    // Base angular velocity expressed in world frame
    const ignition::math::Vector3d &angularVelocity = baseLink_->WorldAngularVel();
    const Eigen::Vector3d angularVelocityWorld(angularVelocity.X(), angularVelocity.Y(), angularVelocity.Z());

    // Base linear velocity expressed in world frame
    const ignition::math::Vector3d &linearVelocity = baseLink_->WorldLinearVel();
    const Eigen::Vector3d linearVelocityWorld(linearVelocity.X(), linearVelocity.Y(), linearVelocity.Z());

    // Joint velocities
    std::vector<double> jointVelocities(joints_.size());
    for (int i = 0; i < joints_.size(); ++i) {
        jointVelocities[i] = joints_[i]->GetVelocity(0);
    }

    // Put everything into an RbdState message
    tbai_msgs::RbdState message;  // TODO(lnotspotl): Room for optimization here

    // Base orientation - ZYX = {yaw, pitch, roll}
    message.rbd_state[0] = rpy[2];  // yaw
    message.rbd_state[1] = rpy[1];  // pitch
    message.rbd_state[2] = rpy[0];  // roll

    // Base position
    message.rbd_state[3] = basePosition[0];
    message.rbd_state[4] = basePosition[1];
    message.rbd_state[5] = basePosition[2];

    // Joint angles
    for (int i = 0; i < jointAngles.size(); ++i) {
        message.rbd_state[6 + i] = jointAngles[i];
    }

    // Base angular velocity
    message.rbd_state[18] = angularVelocityWorld[0];
    message.rbd_state[19] = angularVelocityWorld[1];
    message.rbd_state[20] = angularVelocityWorld[2];

    // Base linear velocity
    message.rbd_state[21] = linearVelocityWorld[0];
    message.rbd_state[22] = linearVelocityWorld[1];
    message.rbd_state[23] = linearVelocityWorld[2];

    // Joint velocities
    for (int i = 0; i < jointVelocities.size(); ++i) {
        message.rbd_state[24 + i] = jointVelocities[i];
    }

    // Publish message
    statePublisher_.publish(message);

    // Update last publish time
    lastPublishTime_ = currentTime;
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
Eigen::Vector3d eulerXYZFromRotationMatrix(const Eigen::Matrix3d &R, double lastYaw) {
    Eigen::Vector3d eulerXYZ = R.eulerAngles(0, 1, 2);
    ocs2::makeEulerAnglesUnique(eulerXYZ);
    eulerXYZ.z() = ocs2::moduloAngleWithReference(eulerXYZ.z(), lastYaw);
    return eulerXYZ;
}

GZ_REGISTER_MODEL_PLUGIN(StateEstimator);

}  // namespace gazebo
