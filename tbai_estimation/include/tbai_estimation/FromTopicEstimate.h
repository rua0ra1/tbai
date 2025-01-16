//
// Created by Ram Charan Teja Thota on 13 January 2025.
// ref from : https://github.com/qiayuanl/legged_control
//

#include "tbai_estimation/StateEstimateBase.h"
#include <realtime_tools/realtime_buffer.h>

#pragma once
namespace tbai {
namespace estimation {
using namespace ocs2;

class FromTopicStateEstimate : public StateEstimateBase {
   public:
    FromTopicStateEstimate(PinocchioInterface pinocchioInterface, CentroidalModelInfo info,
                           const PinocchioEndEffectorKinematics &eeKinematics);

    void updateImu(const Eigen::Quaternion<scalar_t> &quat, const vector3_t &angularVelLocal,
                   const vector3_t &linearAccelLocal, const matrix3_t &orientationCovariance,
                   const matrix3_t &angularVelCovariance, const matrix3_t &linearAccelCovariance) override{};

    vector_t update(const ros::Time &time, const ros::Duration &period) override;

   private:
    void callback(const nav_msgs::Odometry::ConstPtr &msg);

    ros::Subscriber sub_;
    realtime_tools::RealtimeBuffer<nav_msgs::Odometry> buffer_;
};

}  // namespace estimation

}  // namespace tbai