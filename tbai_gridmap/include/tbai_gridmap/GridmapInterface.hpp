#pragma once

#include <memory>
#include <string>

#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <ros/callback_queue.h>
#include <tbai_core/Types.hpp>

namespace tbai {
namespace gridmap {

class GridmapInterface {
   public:
    /**
     * @brief Construct a new Gridmap Interface object
     *
     * @param nh : ROS node handle
     * @param topic : ROS topic to subscribe to that publishes Gridmap messages
     */
    GridmapInterface(ros::NodeHandle &nh, const std::string &topic);

    /**
     * @brief Check if first Gridmap message has been received
     *
     * @return true if initialized
     * @return false otherwise
     */
    inline bool isInitialized() { return mapPtr_.get() != nullptr; }

    /**
     * @brief Wait until first Gridmap message has been received
     *
     */
    void waitTillInitialized();

    /**
     * @brief Return the height at a given position x, y
     *
     * @param x : x coordinate
     * @param y : y coordinate
     * @return scalar_t : height at position x, y
     */
    inline scalar_t atPosition(scalar_t x, scalar_t y) {
        grid_map::Position position(x, y);
        if (!(mapPtr_->isInside(position))) {
            std::cerr << "Position " << x << " " << y << " is outside the map. Returning 0.0" << std::endl;
            return 0.0;
        }
        auto height = mapPtr_->atPosition("elevation_inpainted", position);
        if (std::isnan(height) || std::isinf(height)) {
            std::cerr << "NAN or inf at position " << x << " " << y << " " << height << ". Replacing with 0.0"
                      << std::endl;
            return 0.0;
        }
        return height;
    }

    /**
     * @brief Sample the gridmap at positions given by the input matrix
     *
     * @param sampled : 3xN matrix where each column is a position to sample, first row is x (input), second row is y
     * (input), third row is height (output)
     */
    void atPositions(matrix_t &sampled);

   private:
    /** Gridmap message callback*/
    void callback(const grid_map_msgs::GridMap &msg);

    /** Gridmap message subscriber */
    ros::Subscriber subscriber_;

    /** Pointer to latest gridmap */
    std::unique_ptr<grid_map::GridMap> mapPtr_;
};

/**
 * @brief Get the Gridmap Interface Unique object, initialize with values from a config file
 *
 * @return std::unique_ptr<GridmapInterface> : unique pointer to GridmapInterface object
 */
std::unique_ptr<GridmapInterface> getGridmapInterfaceUnique();

/**
 * @brief Get the Gridmap Interface Shared object, initialize with values from a config file
 *
 * @return std::shared_ptr<GridmapInterface> : shared pointer to GridmapInterface object
 */
std::shared_ptr<GridmapInterface> getGridmapInterfaceShared();

}  // namespace gridmap
}  // namespace tbai
