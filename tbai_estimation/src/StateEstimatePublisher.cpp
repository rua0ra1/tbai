#include <tbai_estimation/StateEstimatePublisher.h>

namespace tbai {
namespace estimation {
StateEstimatePublisher::StateEstimatePublisher(ros::NodeHandle &nh, ros::NodeHandle &pnh_){
    // SubScribers
    auto config = tbai::core::YamlConfig::fromRosParam("/tbai_config_path");
    //Joint state subscriber
    auto stateTopic = config.get<std::string>("JointStatesTopic");
    joint_sub_.subscribe(nh, stateTopic, 10);

    // contact flag subscriber
    auto contactTopic = config.get<std::string>("ContactFlagTopic");
    contact_sub_.subscribe(nh,contactTopic,10);

    // Synchronizer
    sync_.reset(new Sync(SyncPolicy(10), joint_sub_, contact_sub_));
    sync_->registerCallback(boost::bind(&StateEstimatePublisher::synCallback, this, _1, _2));

    //imu subscriber
    auto imuTopic = config.get<std::string>("imuTopic");
    imu_subscriber_ = nh.subscribe<sensor_msgs::Imu>(imuTopic, 1, &StateEstimatePublisher::imuCallback, this);
}

void StateEstimatePublisher::synCallback(const tbai_msgs::JointStatesConstPtr &joint_states_ptr,
                     const tbai_msgs::ContactsConstPtr &contact_flags_ptr){

    ROS_INFO_STREAM("-----synchrozing call back is working---");

    updated_joint_states_.header.stamp=ros::Time::now();

    for(size_t i = 0; i < joint_states_ptr->joints_position.size(); i++)
    {
        updated_joint_states_.joints_position[i]=joint_states_ptr->joints_position[i];
        updated_joint_states_.joints_velocity[i]=joint_states_ptr->joints_velocity[i];
    }

    updated_contact_flags_.header.stamp=ros::Time::now();

    for(size_t i = 0; i < contact_flags_ptr->contact_flags.size(); i++)
    {
        updated_contact_flags_.contact_flags[i]=contact_flags_ptr->contact_flags[i];
    }

}


void StateEstimatePublisher::imuCallback(const sensor_msgs::ImuConstPtr& msg)
{
  updated_imu_ = msg;
}



}  // namespace estimation

}  // namespace tbai
