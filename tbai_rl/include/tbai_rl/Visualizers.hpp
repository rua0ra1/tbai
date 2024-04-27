

#include <memory>

#include <robot_state_publisher/robot_state_publisher.h>
#include <ros/ros.h>
#include <tbai_core/config/YamlConfig.hpp>
#include <tbai_rl/State.hpp>
#include <tf/transform_broadcaster.h>
#include <torch/script.h>

namespace tbai {
namespace rl {

using namespace torch::indexing;

class StateVisualizer {
   public:
    /**
     * @brief Construct a new State Visualizer object
     *
     */
    StateVisualizer();

    /**
     * @brief Publish odom->base transform and joint angles
     *
     * @param state : Current state
     */
    void visualize(const State &state);

   private:
    /** Publish odom->base transform */
    void publishOdomTransform(const ros::Time &timeStamp, const State &state);

    /** Publish joint angles via a robot_state_publisher */
    void publishJointAngles(const ros::Time &timeStamp, const State &state);

    /** Odom frame name */
    std::string odomFrame_;

    /** Base frame name */
    std::string baseFrame_;

    /** List of joint names - must be in the same order as in State*/
    std::vector<std::string> jointNames_;

    /** Helper classes for state visualization */
    std::unique_ptr<robot_state_publisher::RobotStatePublisher> robotStatePublisherPtr_;
    tf::TransformBroadcaster tfBroadcaster_;
};

class HeightsReconstructedVisualizer {
   public:
    /**
     * @brief Construct a new Heights Reconstructed Visualizer object
     *
     */
    HeightsReconstructedVisualizer();

    /**
     * @brief Visualize sampled and reconstructed height samples
     *
     * @param state : Current state
     * @param sampled : 3xN matrix, where each column is a point in 3D space
     * @param nnPointsReconstructed : reconstructed tensor from the neural network
     */
    void visualize(const State &state, const matrix_t &sampled, const at::Tensor &nnPointsReconstructed);

   private:
    /** Publish 3d points as RViz markers */
    void publishMarkers(const ros::Time &timeStamp, const matrix_t &sampled, const std::array<float, 3> &rgb,
                        const std::string &markerNamePrefix, std::function<scalar_t(size_t)> heightFunction);

    /** Odom frame name */
    std::string odomFrame_;

    /** ROS publisher for markers */
    ros::Publisher markerPublisher_;

    /** Whether or not the robot is blind */
    bool blind_;
};

}  // namespace rl
}  // namespace tbai