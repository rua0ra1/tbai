#ifndef TBAI_GRIDMAP_INCLUDE_TBAI_GRIDMAP_GRIDMAPINTERFACE_HPP_
#define TBAI_GRIDMAP_INCLUDE_TBAI_GRIDMAP_GRIDMAPINTERFACE_HPP_

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
    GridmapInterface(ros::NodeHandle &nh, const std::string &topic);
    inline bool isInitialized() { return initialized_; }
    void waitTillInitialized();

    inline scalar_t atPosition(scalar_t x, scalar_t y) {
        grid_map::Position position(x, y);
        if (!(mapPtr_->isInside(position))) {
            return 0.0;
        }
        auto height = mapPtr_->atPosition("elevation_inpainted", position);
        if (std::isnan(height) || std::isinf(height)) {
            std::cerr << "NAN or inf at position " << x << " " << y << " " << height << std::endl;
        }
        return height;
    }

    void atPositions(matrix_t &sampled);

   private:
    void callback(const grid_map_msgs::GridMap &msg);

    ros::Subscriber subscriber_;
    std::unique_ptr<grid_map::GridMap> mapPtr_;
    bool initialized_ = false;
};

std::unique_ptr<GridmapInterface> getGridmapInterfaceUnique();

std::shared_ptr<GridmapInterface> getGridmapInterfaceShared();

}  // namespace gridmap
}  // namespace tbai

#endif  // TBAI_GRIDMAP_INCLUDE_TBAI_GRIDMAP_GRIDMAPINTERFACE_HPP_
