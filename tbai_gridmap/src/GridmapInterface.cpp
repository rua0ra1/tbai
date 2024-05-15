#include "tbai_gridmap/GridmapInterface.hpp"

#include <numeric>

#include <tbai_core/Asserts.hpp>
#include <tbai_core/config/YamlConfig.hpp>

namespace tbai {
namespace gridmap {

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
GridmapInterface::GridmapInterface(ros::NodeHandle &nh, const std::string &topic, const std::string &layer)
    : layer_(layer) {
    subscriber_ = nh.subscribe(topic, 1, &GridmapInterface::callback, this);
}

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
void GridmapInterface::atPositions(matrix_t &sampled) {
    TBAI_ASSERT(isInitialized(), "Gridmap interface must be initialized. Call waitTillInitialized() first.");
    TBAI_ASSERT(sampled.rows() == 3, "Sampled matrix must have 3 rows.");
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
    std::vector<std::string> layers = {layer_};
    grid_map::GridMapRosConverter::fromMessage(msg, *mapPtr, layers, false, false);

    // Swap terrain map pointers
    mapPtr_.swap(mapPtr);
}

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
void GridmapInterface::waitTillInitialized() {
    while (true) {
        ros::spinOnce();
        if (isInitialized()) break;
        ROS_INFO("[GridmapInterface] Waiting for gridmap to be initialized.");
        ros::Duration(0.05).sleep();
    }
}

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
std::unique_ptr<GridmapInterface> getGridmapInterfaceUnique() {
    ros::NodeHandle nh;
    auto topic = tbai::core::fromRosConfig<std::string>("gridmap_topic");
    auto layer = tbai::core::fromRosConfig<std::string>("gridmap_layer");

    return std::unique_ptr<GridmapInterface>(new GridmapInterface(nh, topic, layer));
}

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
std::shared_ptr<GridmapInterface> getGridmapInterfaceShared() {
    return std::shared_ptr<GridmapInterface>(getGridmapInterfaceUnique().release());
}

}  // namespace gridmap
}  // namespace tbai
