#include "tbai_gazebo/StatePublisher.hpp"

#include <functional>
#include <string>

#include <Eigen/Geometry>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <tbai_core/config/YamlConfig.hpp>
#include <tbai_msgs/RbdState.h>

namespace gazebo {

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void StatePublisher::Load(physics::ModelPtr robot, sdf::ElementPtr sdf) {
    ROS_INFO_STREAM("[StatePublisher] Loading StatePublisher plugin");
    // set Gazebo callback function
    updateConnection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&StatePublisher::OnUpdate, this));

    this->robot_ = robot;
    auto config = tbai::core::YamlConfig::fromRosParam("/tbai_config_path");

    ros::NodeHandle nh;
    auto stateTopic = config.get<std::string>("state_topic");
    statePublisher_ = nh.advertise<tbai_msgs::RbdState>(stateTopic, 2);

    auto base = config.get<std::string>("base_name");
    baseLinkPtr_ = robot->GetChildLink(base);

    // get joints; ignore 'universe' and 'root_joint'
    auto jointNames = config.get<std::vector<std::string>>("joint_names");
    for (int i = 0; i < jointNames.size(); ++i) {
        joints_.push_back(robot->GetJoint(jointNames[i]));
    }

    // initialize last publish time
    lastSimTime_ = robot->GetWorld()->SimTime();

    rate_ = config.get<double>("state_publisher/update_rate");
    period_ = 1.0 / rate_;

    ROS_INFO_STREAM("[StatePublisher] Loaded StatePublisher plugin");
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void StatePublisher::OnUpdate() {
  // Get current time
  const common::Time currentTime = robot_->GetWorld()->SimTime();
  const double dt = (currentTime - lastSimTime_).Double();

  // Check if update is needed
  if (dt < period_) {
    return;
  }

  // Base orientation - ZYX Euler angles
  const ignition::math::Pose3d &basePose = baseLinkPtr_->WorldPose();
  Eigen::Quaternion<double> baseQuaternion(
      basePose.Rot().W(), basePose.Rot().X(), basePose.Rot().Y(),
      basePose.Rot().Z());

  // Base position - XYZ world coordinates
  const Eigen::Vector3d basePosition(basePose.Pos().X(), basePose.Pos().Y(),
                                     basePose.Pos().Z());

  auto baseOrientationMat = baseQuaternion.toRotationMatrix();
  if (firstUpdate_) {
    lastBaseOrientationMat_ = baseOrientationMat;
    lastBasePosition_ = basePosition;
    firstUpdate_ = false;
  }
  // Joint angles
  std::vector<double> jointAngles(joints_.size());
  for (int i = 0; i < joints_.size(); ++i) {
    jointAngles[i] = joints_[i]->Position(0);
  }

  // Base angular velocity expressed in world frame
  const ignition::math::Vector3d &angularVelocity =
      baseLinkPtr_->WorldAngularVel();
  const Eigen::Vector3d angularVelocityWorld =
      mat2aa(baseOrientationMat * lastBaseOrientationMat_.transpose()) / dt;

  // Base linear velocity expressed in world frame
  const ignition::math::Vector3d &linearVelocity =
      baseLinkPtr_->WorldLinearVel();
  const Eigen::Vector3d linearVelocityWorld =
      (basePosition - lastBasePosition_) / dt;

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
    tbai::scalar_t current_angle = jointAngles[i];
    tbai::scalar_t last_angle = lastJointAngles_[i];
    tbai::scalar_t velocity = (current_angle - last_angle) / dt;
    jointVelocities[i] = velocity;
    lastJointAngles_[i] = current_angle;
  }

  // Put everything into an RbdState message
  tbai_msgs::RbdState message; // TODO(lnotspotl): Room for optimization here

  // Base position
  message.rbd_state[0] = basePosition[0];
  message.rbd_state[1] = basePosition[1];
  message.rbd_state[2] = basePosition[2];

  // Base orientation
  message.rbd_state[3] = baseQuaternion.x();
  message.rbd_state[4] = baseQuaternion.y();
  message.rbd_state[5] = baseQuaternion.z();
  message.rbd_state[6] = baseQuaternion.w();

  // Joint angles
  for (int i = 0; i < jointAngles.size(); ++i) {
    message.rbd_state[7 + i] = jointAngles[i];
  }

  // Base linear velocity
  message.rbd_state[19] = linearVelocityWorld[0];
  message.rbd_state[20] = linearVelocityWorld[1];
  message.rbd_state[21] = linearVelocityWorld[2];

  // Base angular velocity
  message.rbd_state[22] = angularVelocityWorld[0];
  message.rbd_state[23] = angularVelocityWorld[1];
  message.rbd_state[24] = angularVelocityWorld[2];

  // Joint velocities
  for (int i = 0; i < jointVelocities.size(); ++i) {
    message.rbd_state[25 + i] = jointVelocities[i];
  }

  // Publish message
  statePublisher_.publish(message);

  // Update last publish time
  lastSimTime_ = currentTime;

  lastBaseOrientationMat_ = baseOrientationMat;
  lastBasePosition_ = basePosition;
}

tbai::vector3_t StatePublisher::mat2aa(const tbai::matrix3_t &R) {
  tbai::angleaxis_t aa(R);
  return aa.axis() * aa.angle();
}

GZ_REGISTER_MODEL_PLUGIN(StatePublisher);

}  // namespace gazebo
