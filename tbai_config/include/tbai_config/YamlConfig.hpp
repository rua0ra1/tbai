#ifndef TBAI_CONFIG_INCLUDE_TBAI_CONFIG_YAMLCONFIG_HPP_
#define TBAI_CONFIG_INCLUDE_TBAI_CONFIG_YAMLCONFIG_HPP_

#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace tbai {
namespace config {

class YamlConfig {
   public:
    /**
     * @brief Construct a new Yaml Config object
     *
     * @param configPath : Path to yaml config file
     * @param delim  : config path delimiter
     */
    YamlConfig(const std::string &configPath, const char delim = '/');

    /**
     * @brief Construct YamlConfig with config path specified by a ROS parameter
     *
     * @param pathParam : ROS parameter name containing config's path
     * @param delim : config path delimiter
     * @return YamlConfig object
     */
    static YamlConfig fromRosParam(const std::string &pathParam, const char delim = '/');

    /**
     * @brief Get value from config
     *
     * @tparam T : Type of value to get
     * @param path : Path to value in config
     * @return T : Value from config
     */
    template <typename T>
    T get(const std::string &path) const;

   private:
    /** Split config path: a.b.c -> {'a', 'b', 'c'} if delim_ == '.' */
    std::vector<std::string> split(const std::string &s) const;

    /** Get value from a YAML node */
    template <typename T>
    T parseNode(const YAML::Node &node) const;

    /** Get YAML node specified by its path in config*/
    template <typename T>
    T traverse(const YAML::Node &node, const std::string &key) const;

    std::string configPath_;
    char delim_;
};

}  // namespace config
}  // namespace tbai

#include "tbai_config/implementation/YamlConfig.hpp"

#endif  // TBAI_CONFIG_INCLUDE_TBAI_CONFIG_YAMLCONFIG_HPP_
