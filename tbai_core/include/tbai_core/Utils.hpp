#pragma once

#include <ros/ros.h>
#include <tbai_core/Types.hpp>

namespace tbai {
namespace core {

inline void setEpochStart() {
    ros::NodeHandle().setParam("epoch_start", ros::Time::now().toSec());
}

inline bool isEpochStartSet() {
    return ros::NodeHandle().hasParam("epoch_start");
}

inline scalar_t getEpochStart() {
    double epochStart;
    while (ros::ok()) {
        if (ros::param::get("epoch_start", epochStart)) {
            return static_cast<scalar_t>(epochStart);
        }
        ros::Duration(0.05).sleep();
    }
    ROS_ERROR("Failed to get epoch start time");
    return static_cast<scalar_t>(-1.0);
}

}  // namespace core
}  // namespace tbai
