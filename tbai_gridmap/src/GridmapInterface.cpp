#include "tbai_gridmap/GridmapInterface.hpp"

namespace tbai {
namespace gridmap {

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
GridmapInterface::GridmapInterface(ros::NodeHandle &nh, const std::string &topic, const std::string &layer)
    : initialized_(false), layer_(layer) {
    subscriber_ = nh.subscribe(topic, 1, &GridmapInterface::callback, this);
    callbackQueueInterfaceRawPtr_ = dynamic_cast<ros::CallbackQueue *>(nh.getCallbackQueue());
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
    grid_map::GridMapRosConverter::fromMessage(msg, map_);
    initialized_ = true;
}

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
void GridmapInterface::waitTillInitialized() {
    while (!initialized_) {
        ROS_INFO("[GridmapInterface] Waiting for gridmap to be initialized.");
        ros::Duration(0.2).sleep();
        spinOnce();
    }
}

}  // namespace gridmap
}  // namespace tbai
