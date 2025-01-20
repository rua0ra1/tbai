#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/fwd.hpp>  // forward declarations must be included first.
#include <tbai_core/Throws.hpp>
#include <tbai_core/Utils.hpp>
#include <tbai_estimation/StateEstimatePublisher.h>

namespace tbai {
namespace estimation {
StateEstimatePublisher::StateEstimatePublisher(ros::NodeHandle &nh, ros::NodeHandle &pnh) {
    // SubScribers
    auto config = tbai::core::YamlConfig::fromRosParam("/tbai_config_path");
    // Joint state subscriber
    auto stateTopic = config.get<std::string>("JointStatesTopic");
    joint_sub_.subscribe(nh, stateTopic, 5);

    // contact flag subscriber
    auto contactTopic = config.get<std::string>("ContactFlagTopic");
    contact_sub_.subscribe(nh, contactTopic, 5);

    // Synchronizer
    sync_.reset(new Sync(SyncPolicy(10), joint_sub_, contact_sub_));
    sync_->registerCallback(boost::bind(&StateEstimatePublisher::synCallback, this, _1, _2));

    // imu subscriber
    auto imuTopic = config.get<std::string>("imuTopic");
    imu_subscriber_ = nh.subscribe<sensor_msgs::Imu>(imuTopic, 1, &StateEstimatePublisher::imuCallback, this);

    // ROS timer that repeatedly calls a callback function at a specified interval.
    state_pub_timer_ = pnh.createTimer(ros::Duration(0.02), &StateEstimatePublisher::publishStateToRBDmsg, this);

    // read the urdf file
    // URDF
    std::string urdfString="/home/ram/personal/icra_quadraped_comptettion/other_useful_quadruped_implementaion/tbai_org/catkin_ws/src/tbai/dependencies/ocs2_robotic_assets/resources/anymal_d/urdf/anymal.urdf";
    //TBAI_ROS_THROW_IF(!nh.getParam("/robot_description", urdfString), "Failed to get parameter /robot_description");
    // setup pinnochio interface
    pinocchioInterfacePtr_ = std::make_unique<PinocchioInterface>(
        centroidal_model::createPinocchioInterface(urdfString, modelSettings_.jointNames));

    // Task settings
    std::string taskfile;
    TBAI_ROS_THROW_IF(!nh.getParam("/task_file", taskfile), "Failed to get parameter /task_settings_file");
    // Frame declarations
    std::string referencefile;
    TBAI_ROS_THROW_IF(!nh.getParam("/reference_file", referencefile),
                      "Failed to get parameter /frame_declaration_file");

    // CentroidalModelInfo
    centroidalModelInfo_ = centroidal_model::createCentroidalModelInfo(
        *pinocchioInterfacePtr_, centroidal_model::loadCentroidalType(taskfile),
        centroidal_model::loadDefaultJointState(pinocchioInterfacePtr_->getModel().nq - 6, referencefile),
        modelSettings_.contactNames3DoF, modelSettings_.contactNames6DoF);

    // endeffector kinematics
    CentroidalModelPinocchioMapping pinocchioMapping(centroidalModelInfo_);
    eeKinematicsPtr_ = std::make_shared<PinocchioEndEffectorKinematics>(*pinocchioInterfacePtr_, pinocchioMapping,
                                                                        modelSettings_.contactNames3DoF);

    // state estimate
    stateEstimate_ =
        std::make_shared<KalmanFilterEstimate>(*pinocchioInterfacePtr_, centroidalModelInfo_, *eeKinematicsPtr_);
}

void StateEstimatePublisher::synCallback(const tbai_msgs::JointStatesConstPtr &joint_states_ptr,
                                         const tbai_msgs::ContactsConstPtr &contact_flags_ptr) {
    updated_joint_states_.header.stamp = ros::Time::now();

    for (size_t i = 0; i < joint_states_ptr->joints_position.size(); i++) {
        updated_joint_states_.joints_position[i] = joint_states_ptr->joints_position[i];
        updated_joint_states_.joints_velocity[i] = joint_states_ptr->joints_velocity[i];
    }

    updated_contact_flags_.header.stamp = ros::Time::now();

    for (size_t i = 0; i < contact_flags_ptr->contact_flags.size(); i++) {
        updated_contact_flags_.contact_flags[i] = contact_flags_ptr->contact_flags[i];
    }

    joint_contact_ready = true;
}

void StateEstimatePublisher::imuCallback(const sensor_msgs::ImuConstPtr &msg) {
    updated_imu_ = msg;
    imu_data_ready = true;
}

void StateEstimatePublisher::publishStateToRBDmsg(const ros::TimerEvent &event) {
    if (imu_data_ready && joint_contact_ready) {
        ROS_INFO_STREAM("-------- loop is running:---------");
        vector_t jointPos(12), jointVel(12);
        contact_flag_t contacts;
        contact_flag_t contactFlag;
        vector3_t angularVel, linearAccel;
        matrix3_t orientationCovariance, angularVelCovariance, linearAccelCovariance;

        for (size_t i = 0; i < 12; ++i) {
            jointPos(i) = updated_joint_states_.joints_position[i];
            jointVel(i) = updated_joint_states_.joints_velocity[i];
        }
        for (size_t i = 0; i < 4; ++i) {
            contactFlag[i] = static_cast<bool>(updated_contact_flags_.contact_flags[i]);
        }
        // update the quaternion angels
        // Extract quaternion components from the ROS Imu message
        scalar_t w = updated_imu_->orientation.w;
        scalar_t x = updated_imu_->orientation.x;
        scalar_t y = updated_imu_->orientation.y;
        scalar_t z = updated_imu_->orientation.z;

        auto quat = Eigen::Quaternion<scalar_t>(w, x, y, z);
        quat.normalize();

        // update linear accelrarion and angular velocity and its respective covariances
        angularVel(0) = updated_imu_->angular_velocity.x;
        angularVel(1) = updated_imu_->angular_velocity.y;
        angularVel(2) = updated_imu_->angular_velocity.z;

        linearAccel(0) = updated_imu_->linear_acceleration.x;
        linearAccel(1) = updated_imu_->linear_acceleration.y;
        linearAccel(2) = updated_imu_->linear_acceleration.z;

        for (size_t i = 0; i < 9; ++i) {
            orientationCovariance(i) = updated_imu_->orientation_covariance[i];
            angularVelCovariance(i) = updated_imu_->angular_velocity_covariance[i];
            linearAccelCovariance(i) = updated_imu_->linear_acceleration_covariance[i];
        }

        stateEstimate_->updateJointStates(jointPos, jointVel);
        stateEstimate_->updateContact(contactFlag);

        stateEstimate_->updateJointStates(jointPos, jointVel);
        stateEstimate_->updateContact(contactFlag);
        stateEstimate_->updateImu(quat, angularVel, linearAccel, orientationCovariance, angularVelCovariance,
                                  linearAccelCovariance);
       auto measuredRbdState_ = stateEstimate_->update(ros::Time::now(), ros::Duration(0.02));

    }
}

}  // namespace estimation

}  // namespace tbai
