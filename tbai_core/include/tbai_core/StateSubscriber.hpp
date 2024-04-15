#ifndef TBAI_CORE_INCLUDE_TBAI_CORE_STATESUBSCRIBER_HPP_
#define TBAI_CORE_INCLUDE_TBAI_CORE_STATESUBSCRIBER_HPP_

#include <string>

#include "tbai_core/Types.hpp"
#include "tbai_msgs/RbdState.h"
#include <ros/ros.h>

namespace tbai {
namespace core {

class StateSubscriber {
   public:
    /**
     * @brief Construct a new State Subscriber object
     *
     * @param nh : ROS node handle
     * @param stateTopic : ROS topic name for state messages
     */
    StateSubscriber(ros::NodeHandle &nh, const std::string &stateTopic);  // NOLINT

    /**
     * @brief Wait until the first state message is received
     *
     */
    void waitTillInitialized();

    /**
     * @brief Get latest Rbd state
     *
     * @return const vector_t& : latest rbd state
     */
    const vector_t &getLatestRbdState();

   private:
    /** State message callback */
    void stateMessageCallback(const tbai_msgs::RbdState::Ptr &msg);

    /** Convert state message to vector_t */
    void updateLatestRbdState();

    /** Shared pointer to the latest state message */
    tbai_msgs::RbdState::Ptr stateMessage_;

    /** State message subscriber */
    ros::Subscriber stateSubscriber_;

    /** Whether or not the latest message has been converted to vector_t */
    bool stateReady_ = false;

    /** Latest Rbd state */
    vector_t latestRbdState_;
};

}  // namespace core
}  // namespace tbai

#endif  // TBAI_CORE_INCLUDE_TBAI_CORE_STATESUBSCRIBER_HPP_
