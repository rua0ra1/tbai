// clang-format off
#include <pinocchio/fwd.hpp>
// clang-format on

#include <ros/ros.h>
#include <tbai_core/config/YamlConfig.hpp>
#include <tbai_core/control/CentralController.hpp>
#include <tbai_dtc/DtcController.hpp>
#include <tbai_static/StaticController.hpp>

#include <tbai_core/Utils.hpp>

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "tbai_static");
    const std::string configParam = "/tbai_config_path";

    tbai::core::setEpochStart();

    auto config = tbai::core::YamlConfig::fromRosParam(configParam);
    auto stateTopic = config.get<std::string>("state_topic");
    auto commandTopic = config.get<std::string>("command_topic");
    auto changeControllerTopic = config.get<std::string>("change_controller_topic");

    ros::NodeHandle nh;
    tbai::core::CentralController controller(nh, stateTopic, commandTopic, changeControllerTopic);

    // Add static controller
    controller.addController(
        std::make_unique<tbai::static_::StaticController>(configParam, controller.getStateSubscriberPtr()));
    controller.addController(std::make_unique<tbai::dtc::DtcController>(controller.getStateSubscriberPtr()));

    // Start controller loop
    controller.start();

    return EXIT_SUCCESS;
}
