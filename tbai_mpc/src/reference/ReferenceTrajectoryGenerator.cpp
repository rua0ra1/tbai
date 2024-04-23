#include "tbai_mpc/reference/ReferenceTrajectoryGenerator.hpp"

#include <tbai_core/config/YamlConfig.hpp>
namespace tbai {
namespace mpc {
namespace reference {

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
ReferenceTrajectoryGenerator::ReferenceTrajectoryGenerator(const std::string &targetCommandFile, ros::NodeHandle &nh)
    : firstObservationReceived_(false) {
    defaultJointState_.setZero(12);
    loadSettings(targetCommandFile);

    trajdt_ = tbai::core::fromRosConfig<scalar_t>("mpc_controller/reference_trajectory/traj_dt");
    trajKnots_ = tbai::core::fromRosConfig<size_t>("mpc_controller/reference_trajectory/traj_knots");

    // Setup ROS subscribers
    auto observationTopic =
        tbai::core::fromRosConfig<std::string>("mpc_controller/reference_trajectory/observation_topic");
    observationSubscriber_ =
        nh.subscribe(observationTopic, 1, &ReferenceTrajectoryGenerator::observationCallback, this);

    auto terrainTopic = tbai::core::fromRosConfig<std::string>("mpc_controller/reference_trajectory/terrain_topic");
    terrainSubscriber_ = nh.subscribe(terrainTopic, 1, &ReferenceTrajectoryGenerator::terrainCallback, this);

    // Setup ROS publishers
    auto referenceTopic = tbai::core::fromRosConfig<std::string>("mpc_controller/reference_trajectory/reference_topic");
    referencePublisher_ = nh.advertise<ocs2_msgs::mpc_target_trajectories>(referenceTopic, 1, false);

    blind_ = tbai::core::fromRosConfig<bool>("mpc_controller/reference_trajectory/blind");
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/

void ReferenceTrajectoryGenerator::publishReferenceTrajectory() {
    if (!firstObservationReceived_) {
        ROS_WARN_THROTTLE(1.0, "No observation received yet. Cannot publish reference trajectory.");
        return;
    }

    // Generate reference trajectory
    ocs2::TargetTrajectories referenceTrajectory = generateReferenceTrajectory(ros::Time::now().toSec());

    // Publish reference trajectory
    referencePublisher_.publish(ocs2::ros_msg_conversions::createTargetTrajectoriesMsg(referenceTrajectory));
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
BaseReferenceHorizon ReferenceTrajectoryGenerator::getBaseReferenceHorizon() {
    return {trajdt_, trajKnots_};
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
BaseReferenceCommand ReferenceTrajectoryGenerator::getBaseReferenceCommand(scalar_t time) {
    const auto velocityCommand = commandControllerPtr_->getVelocityCommand(time);
    return {velocityCommand.velocity_x, velocityCommand.velocity_y, velocityCommand.yaw_rate, comHeight_};
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
BaseReferenceState ReferenceTrajectoryGenerator::getBaseReferenceState() {
    std::lock_guard<std::mutex> lock(observationMutex_);
    scalar_t observationTime = latestObservation_.time;
    Eigen::Vector3d positionInWorld = latestObservation_.state.segment<3>(3);
    Eigen::Vector3d eulerXyz = latestObservation_.state.head<3>();
    return {observationTime, positionInWorld, eulerXyz};
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
TerrainPlane &ReferenceTrajectoryGenerator::getTerrainPlane() {
    return localTerrain_;
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
ocs2::TargetTrajectories ReferenceTrajectoryGenerator::generateReferenceTrajectory(scalar_t time, scalar_t dt) {
    // Get base reference trajectory
    BaseReferenceTrajectory baseReferenceTrajectory;
    if (!terrainMapPtr_) {
        baseReferenceTrajectory = generateExtrapolatedBaseReference(getBaseReferenceHorizon(), getBaseReferenceState(),
                                                                    getBaseReferenceCommand(time), getTerrainPlane());

    } else {
        baseReferenceTrajectory =
            generateExtrapolatedBaseReference(getBaseReferenceHorizon(), getBaseReferenceState(),
                                              getBaseReferenceCommand(time), *terrainMapPtr_, 0.5, 0.3);
    }

    // Generate target trajectory
    ocs2::scalar_array_t desiredTimeTrajectory = std::move(baseReferenceTrajectory.time);
    const size_t N = desiredTimeTrajectory.size();
    ocs2::vector_array_t desiredStateTrajectory(N);
    ocs2::vector_array_t desiredInputTrajectory(N, ocs2::vector_t::Zero(INPUT_DIM));
    for (size_t i = 0; i < N; ++i) {
        ocs2::vector_t state = ocs2::vector_t::Zero(STATE_DIM);

        // base orientation
        state.head<3>() = baseReferenceTrajectory.eulerXyz[i];

        auto Rt = switched_model::rotationMatrixOriginToBase(baseReferenceTrajectory.eulerXyz[i]);

        // base position
        state.segment<3>(3) = baseReferenceTrajectory.positionInWorld[i];

        // base angular velocity
        state.segment<3>(6) = Rt * baseReferenceTrajectory.angularVelocityInWorld[i];

        // base linear velocity
        state.segment<3>(9) = Rt * baseReferenceTrajectory.linearVelocityInWorld[i];

        // joint angles
        state.segment<12>(12) = defaultJointState_;

        desiredStateTrajectory[i] = std::move(state);
    }

    return ocs2::TargetTrajectories(std::move(desiredTimeTrajectory), std::move(desiredStateTrajectory),
                                    std::move(desiredInputTrajectory));
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void ReferenceTrajectoryGenerator::loadSettings(const std::string &targetCommandFile) {
    // Load target COM height
    ocs2::loadData::loadCppDataType<scalar_t>(targetCommandFile, "comHeight", comHeight_);

    // Load default joint angles
    ocs2::loadData::loadEigenMatrix(targetCommandFile, "defaultJointState", defaultJointState_);
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void ReferenceTrajectoryGenerator::observationCallback(const ocs2_msgs::mpc_observation::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(observationMutex_);
    latestObservation_ = ocs2::ros_msg_conversions::readObservationMsg(*msg);
    firstObservationReceived_ = true;
}

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void ReferenceTrajectoryGenerator::terrainCallback(const grid_map_msgs::GridMap &msg) {
    if (!blind_) {
        // Convert ROS message to grid map
        std::unique_ptr<grid_map::GridMap> mapPtr(new grid_map::GridMap);
        std::vector<std::string> layers = {"smooth_planar"};
        grid_map::GridMapRosConverter::fromMessage(msg, *mapPtr, layers, false, false);

        // Swap terrain map pointers
        terrainMapPtr_.swap(mapPtr);
    }
}

}  // namespace reference
}  // namespace mpc
}  // namespace tbai