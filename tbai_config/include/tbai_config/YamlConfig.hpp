#ifndef TBAI_CONFIG_INCLUDE_TBAI_CONFIG_YAMLCONFIG_HPP_
#define TBAI_CONFIG_INCLUDE_TBAI_CONFIG_YAMLCONFIG_HPP_

#include <yaml-cpp/yaml.h>

#include <string>

class YamlConfig {
   public:
    /**
     * @brief Construct a new Yaml Config object
     *
     * @param configPath : Path to yaml config file
     * @param delim  : config path delimiter
     */
    YamlConfig(const std::string &configPath, const std::string &delim = "/");

    /**
     * @brief Construct YamlConfig with config path specified by a ROS parameter
     *
     * @param pathParam : ROS parameter name containing config's path
     * @param delim : config path delimiter
     * @return YamlConfig object
     */
    static YamlConfig fromRosParam(const std::string &pathParam, const std::string &delim = "/");

   private:
    std::string configPath_;
    std::string delim_;
};

#include "tbai_config/implementation/YamlConfig.hpp"

#endif  // TBAI_CONFIG_INCLUDE_TBAI_CONFIG_YAMLCONFIG_HPP_
