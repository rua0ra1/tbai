#ifndef TBAI_CORE_INCLUDE_TBAI_CORE_CENTRALCONTROLLER_HPP_
#define TBAI_CORE_INCLUDE_TBAI_CORE_CENTRALCONTROLLER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "tbai_core/Controller.hpp"
#include "tbai_core/StateSubscriber.hpp"
#include "tbai_core/Types.hpp"
#include <std_msgs/String.h>

namespace tbai {
namespace core {

class CentralController {
   public:
    CentralController(const std::string &configParam);

    void start();

    void addController(std::unique_ptr<Controller> controller, bool makeActive = false);

    std::shared_ptr<StateSubscriber> getStateSubscriber();

   private:
    void step(scalar_t currentTime);
    void visualize();

    void changeControllerCallback(const std_msgs::String::ConstPtr &msg);

    std::vector<std::unique_ptr<Controller>> controllers_;
    Controller *activeController_;

    ros::Rate loopRate_;
};

}  // namespace core
}  // namespace tbai

#endif  // TBAI_CORE_INCLUDE_TBAI_CORE_CENTRALCONTROLLER_HPP_
