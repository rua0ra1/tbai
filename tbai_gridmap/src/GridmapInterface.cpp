#include "tbai_gridmap/GridmapInterface.hpp"

#include <numeric>

#include <tbai_core/config/YamlConfig.hpp>

namespace tbai {
namespace gridmap {

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
GridmapInterface::GridmapInterface(ros::NodeHandle &nh, const std::string &topic) : initialized_(false) {
    subscriber_ = nh.subscribe(topic, 1, &GridmapInterface::callback, this);
}

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
void GridmapInterface::atPositions(matrix_t &sampled) {
    for (int i = 0; i < sampled.cols(); ++i) {
        scalar_t x = sampled(0, i);
        scalar_t y = sampled(1, i);
        sampled(2, i) = atPosition(x, y);
    }
}

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
void GridmapInterface::callback(const grid_map_msgs::GridMap &msg) {
    // Convert ROS message to grid map
    std::unique_ptr<grid_map::GridMap> mapPtr(new grid_map::GridMap);
    std::vector<std::string> layers = {"elevation_inpainted"};
    grid_map::GridMapRosConverter::fromMessage(msg, *mapPtr, layers, false, false);

    // Swap terrain map pointers
    mapPtr_.swap(mapPtr);
    initialized_ = true;
}

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
void GridmapInterface::waitTillInitialized() {
    while (!initialized_) {
        ROS_INFO("[Bobnet gridmap] Waiting for gridmap to be initialized.");
        ros::Duration(0.5).sleep();
        ros::spinOnce();
    }
}

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
std::unique_ptr<GridmapInterface> getGridmapInterfaceUnique() {
    ros::NodeHandle nh;
    auto topic = tbai::core::fromRosConfig<std::string>("gridmap_topic");

    return std::unique_ptr<GridmapInterface>(new GridmapInterface(nh, topic));
}

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
std::shared_ptr<GridmapInterface> getGridmapInterfaceShared() {
    return std::shared_ptr<GridmapInterface>(getGridmapInterfaceUnique().release());
}

}  // namespace gridmap
}  // namespace tbai
