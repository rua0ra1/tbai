//
// Created by Ram Charan Teja Thota on 16 January 2025.
//
//

#pragma once

#include <ros/ros.h>

#include <Eigen/Dense>
#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/Imu.h>
#include <tbai_msgs/Contacts.h>
#include <tbai_msgs/JointStates.h>
#include <tbai_msgs/RbdState.h>

#include <tbai_core/config/YamlConfig.hpp>


namespace tbai {
namespace estimation {

class StateEstimatePublisher {
   public:
    StateEstimatePublisher(ros::NodeHandle &nh, ros::NodeHandle &pnh_);

   private:
    // Subscribers and synchronizer jointstates and contacts flags
    message_filters::Subscriber<tbai_msgs::Contacts> contact_sub_;
    message_filters::Subscriber<tbai_msgs::JointStates> joint_sub_;
    typedef message_filters::sync_policies::ApproximateTime<tbai_msgs::JointStates, tbai_msgs::Contacts> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Sync;
    std::shared_ptr<Sync> sync_;
    void synCallback(const tbai_msgs::JointStatesConstPtr &joint_states_ptr,
                     const tbai_msgs::ContactsConstPtr &contact_flags_ptr);


    tbai_msgs::JointStates updated_joint_states_;
    tbai_msgs::Contacts updated_contact_flags_;

   

    // imu subcriber
    ros::Subscriber imu_subscriber_;
    void imuCallback(const sensor_msgs::ImuConstPtr& imu_msg);
    sensor_msgs::ImuConstPtr updated_imu_;

     // estimated state publisher ( as rbdstate msg)
    ros::Publisher tbai_rbd_state_pub_;
};

}  // namespace estimation
}  // namespace tbai
