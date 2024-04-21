#ifndef TBAI_GRIDMAP_INCLUDE_TBAI_GRIDMAP_GRIDMAPINTERFACE_HPP_
#define TBAI_GRIDMAP_INCLUDE_TBAI_GRIDMAP_GRIDMAPINTERFACE_HPP_

#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <ros/callback_queue.h>
#include <tbai_core/Types.hpp>

namespace tbai {
namespace gridmap {

class GridmapInterface {
   public:
    GridmapInterface(ros::NodeHandle &nh, const std::string &topic, const std::string &layer);
    inline bool isInitialized() { return initialized_; }
    void waitTillInitialized();

    inline scalar_t atPosition(scalar_t x, scalar_t y) {
        grid_map::Position position(x, y);
        if (!map_.isInside(position)) {
            return 0.0;
        }
        return map_.atPosition(layer_, position);
    }

    const grid_map::GridMap &getMap() { return map_; }

    void atPositions(matrix_t &sampled);

   private:
    void callback(const grid_map_msgs::GridMap &msg);

    inline void spinOnce() { callbackQueueInterfaceRawPtr_->callAvailable(ros::WallDuration()); }

    ros::CallbackQueue *callbackQueueInterfaceRawPtr_;

    ros::Subscriber subscriber_;
    grid_map::GridMap map_;
    std::string layer_;
    bool initialized_ = false;

    void generateSamplingPositions();
};

}  // namespace gridmap
}  // namespace tbai

#endif  // TBAI_GRIDMAP_INCLUDE_TBAI_GRIDMAP_GRIDMAPINTERFACE_HPP_