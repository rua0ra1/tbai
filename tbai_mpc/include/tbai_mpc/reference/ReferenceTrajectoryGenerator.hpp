#ifndef TBAI_MPC_INCLUDE_TBAI_MPC_REFERENCE_REFERENCETRAJECTORYGENERATOR_HPP_
#define TBAI_MPC_INCLUDE_TBAI_MPC_REFERENCE_REFERENCETRAJECTORYGENERATOR_HPP_

#include <mutex>
#include <string>

#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <ocs2_anymal_commands/ReferenceExtrapolation.h>
#include <ocs2_core/Types.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_msgs/mpc_observation.h>
#include <ocs2_ros_interfaces/common/RosMsgConversions.h>
#include <ros/ros.h>

#include <tbai_reference/ReferenceVelocityGenerator.hpp>

namespace tbai {
namespace mpc {
namespace reference {

using namespace switched_model;

using ocs2::scalar_t;
using ocs2::SystemObservation;

class ReferenceTrajectoryGenerator {
   public:
    ReferenceTrajectoryGenerator(const std::string &targetCommandFile, ros::NodeHandle &nh);
    void publishReferenceTrajectory();
    void reset() {
        firstObservationReceived_ = false;
    }

    bool isInitialized() const { return firstObservationReceived_; }

   private:
    BaseReferenceHorizon getBaseReferenceHorizon();
    BaseReferenceCommand getBaseReferenceCommand(scalar_t time);
    BaseReferenceState getBaseReferenceState();
    TerrainPlane &getTerrainPlane();
    ocs2::TargetTrajectories generateReferenceTrajectory(scalar_t time, scalar_t dt);

    void loadSettings(const std::string &targetCommandFile);

    std::unique_ptr<tbai::reference::ReferenceVelocityGenerator> velocityGeneratorPtr_;

    // ROS callbacks
    ros::Subscriber observationSubscriber_;
    void observationCallback(const ocs2_msgs::mpc_observation::ConstPtr &msg);

    ros::Subscriber terrainSubscriber_;
    void terrainCallback(const grid_map_msgs::GridMap &msg);

    ros::Publisher referencePublisher_;

    ocs2::vector_t defaultJointState_;
    TerrainPlane localTerrain_;
    bool firstObservationReceived_;
    scalar_t comHeight_;
    scalar_t trajdt_;   // timestep
    size_t trajKnots_;  // number of timesteps in reference horizon
    std::mutex observationMutex_;
    std::mutex terrainMutex_;
    std::unique_ptr<grid_map::GridMap> terrainMapPtr_;
    SystemObservation latestObservation_;

    bool blind_;
};

std::unique_ptr<ReferenceTrajectoryGenerator> getReferenceTrajectoryGeneratorUnique(ros::NodeHandle &nh);

std::shared_ptr<ReferenceTrajectoryGenerator> getReferenceTrajectoryGeneratorShared(ros::NodeHandle &nh);

}  // namespace reference
}  // namespace mpc
}  // namespace tbai

#endif  // TBAI_MPC_INCLUDE_TBAI_MPC_REFERENCE_REFERENCETRAJECTORYGENERATOR_HPP_
