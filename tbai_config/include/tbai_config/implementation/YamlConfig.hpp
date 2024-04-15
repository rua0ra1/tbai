#ifndef TBAI_CONFIG_INCLUDE_TBAI_CONFIG_IMPLEMENTATION_YAMLCONFIG_HPP_
#define TBAI_CONFIG_INCLUDE_TBAI_CONFIG_IMPLEMENTATION_YAMLCONFIG_HPP_

#include <iostream>
#include <string>

#include <tbai_core/Types.hpp>

namespace tbai {

namespace config {

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
template <typename T>
T YamlConfig::traverse(const YAML::Node &node, const std::string &key) const {
    YAML::Node component(node);
    for (auto &k : split(key)) {
        component = component[k];
    }
    return parseNode<T>(component);
}

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
template <typename T>
T YamlConfig::parseNode(const YAML::Node &node) const {
    return node.as<T>();
}

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
/// \cond  // TODO(lnotspotl): This specialization breaks doxygen
template <>
vector_t YamlConfig::parseNode(const YAML::Node &node) const {
    const size_t len = node.size();
    vector_t output(len);
    for (size_t i = 0; i < len; ++i) {
        output(i) = node[i].as<scalar_t>();
    }
    return output;
}
/// \endcond

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
/// \cond  // TODO(lnotspotl): This specialization breaks doxygen
template <>
matrix_t YamlConfig::parseNode(const YAML::Node &node) const {
    const size_t rows = node.size();
    const size_t cols = node[0].size();
    matrix_t output(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output(i, j) = node[i][j].as<scalar_t>();
        }
    }
    return output;
}
/// \endcond

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/
template <typename T>
T YamlConfig::get(const std::string &path) const {
    YAML::Node config = YAML::LoadFile(configPath_);
    if (performChecks_) {
        YAML::Node configCheck = YAML::Clone(config);
        checkExists(configCheck, path);
    }
    return traverse<T>(config, path);
}

}  // namespace config
}  // namespace tbai

#endif  // TBAI_CONFIG_INCLUDE_TBAI_CONFIG_IMPLEMENTATION_YAMLCONFIG_HPP_
