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
    CentralController(ros::NodeHandle &nh, const std::string &stateTopic, const std::string &commandTopic,
                      const std::string &changeControllerTopic);  // NOLINT

    void start();

    void addController(std::unique_ptr<Controller> controller, bool makeActive = false);

    std::shared_ptr<StateSubscriber> getStateSubscriberPtr();

    inline scalar_t getCurrentTime() const { return (ros::Time::now() - initTime_).toSec(); }

   private:
    void step(scalar_t currentTime);
    inline void visualize() { activeController_->visualize(); }

    void changeControllerCallback(const std_msgs::String::ConstPtr &msg);

    std::vector<std::unique_ptr<Controller>> controllers_;
    Controller *activeController_;
    std::shared_ptr<StateSubscriber> stateSubscriberPtr_;

    ros::Rate loopRate_;
    ros::Time initTime_;

    ros::Publisher commandPublisher_;
    ros::Subscriber changeControllerSubscriber_;
};

}  // namespace core
}  // namespace tbai

#endif  // TBAI_CORE_INCLUDE_TBAI_CORE_CENTRALCONTROLLER_HPP_
