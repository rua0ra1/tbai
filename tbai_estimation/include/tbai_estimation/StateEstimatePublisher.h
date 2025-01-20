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

//#include<tbai_estimation/StateEstimateBase.h>

#include <ocs2_centroidal_model/CentroidalModelInfo.h>
#include <ocs2_legged_robot/common/ModelSettings.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>


#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelRbdConversions.h>
// #include <ocs2_self_collision/PinocchioGeometryInterface.h>

#include<tbai_estimation/StateEstimateBase.h>
#include<tbai_estimation/LinearKalmanFilter.h>







namespace tbai {
namespace estimation {
    using namespace ocs2;
    using namespace legged_robot;

class StateEstimatePublisher {
   public:
    StateEstimatePublisher( ros::NodeHandle &nh,  ros::NodeHandle &pnh);

   private:
    // Subscribers and synchronizer jointstates and contacts flags
    message_filters::Subscriber<tbai_msgs::Contacts> contact_sub_;
    message_filters::Subscriber<tbai_msgs::JointStates> joint_sub_;
    typedef message_filters::sync_policies::ApproximateTime<tbai_msgs::JointStates, tbai_msgs::Contacts> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Sync;
    std::shared_ptr<Sync> sync_;
    void synCallback(const tbai_msgs::JointStatesConstPtr &joint_states_ptr,
                     const tbai_msgs::ContactsConstPtr &contact_flags_ptr);
    bool joint_contact_ready=false;
    


    tbai_msgs::JointStates updated_joint_states_;
    tbai_msgs::Contacts updated_contact_flags_;

   

    // imu subcriber
    ros::Subscriber imu_subscriber_;
    void imuCallback(const sensor_msgs::ImuConstPtr& imu_msg);
    sensor_msgs::ImuConstPtr updated_imu_;
    bool imu_data_ready=false;


     // estimated state publisher ( as rbdstate msg)
    ros::Publisher tbai_rbd_state_pub_;
    // state estimation
    std::shared_ptr<StateEstimateBase> stateEstimate_;
    std::shared_ptr<CentroidalModelRbdConversions> rbdConversions_;

    // ROS timer that repeatedly calls a callback function at a specified interval.
    ros::Timer state_pub_timer_;
    void publishStateToRBDmsg(const ros::TimerEvent& event);

    // stateestimate base variable
    // std::shared_ptr<StateEstimateBase> stateEstimate_;

    std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr_;
    CentroidalModelInfo centroidalModelInfo_;
    ModelSettings modelSettings_;
    std::shared_ptr<PinocchioEndEffectorKinematics> eeKinematicsPtr_;


};

}  // namespace estimation
}  // namespacetbai
