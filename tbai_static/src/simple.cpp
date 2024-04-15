#include <iostream>

#include "tbai_static/StaticController.hpp"
#include <ros/ros.h>
#include <tbai_config/YamlConfig.hpp>
#include <tbai_core/CentralController.hpp>

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "tbai_static");
    const std::string configParam = "/tbai_config_path";

    auto config = tbai::config::YamlConfig::fromRosParam(configParam);
    auto stateTopic = config.get<std::string>("state_topic");
    auto commandTopic = config.get<std::string>("command_topic");
    auto changeControllerTopic = config.get<std::string>("change_controller_topic");

    ros::NodeHandle nh;
    tbai::core::CentralController controller(nh, stateTopic, commandTopic, changeControllerTopic);
    controller.addController(std::unique_ptr<tbai::core::Controller>(
        new tbai::static_::StaticController(configParam, controller.getStateSubscriberPtr(), 0.0)));

    controller.start();

    return EXIT_SUCCESS;
}
