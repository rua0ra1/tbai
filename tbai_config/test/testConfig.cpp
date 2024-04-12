#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "tbai_config/YamlConfig.hpp"

#define DUMMY_CONFIG_PATH "./test_config_abc123.yaml"

class YamlConfigTest : public ::testing::Test {
   protected:
    static void SetUpTestSuite() {
        const std::string configPath = DUMMY_CONFIG_PATH;
        std::ofstream file(configPath);
        std::string content = "a:\n  b: hello\n  c: 1\n  d: 3.14\n  e:\n    f: 28";
        file << content;
        file.close();
    }

    static void TearDownTestSuite() {
        const std::string configPath = DUMMY_CONFIG_PATH;
        std::remove(configPath.c_str());
    }
};

TEST_F(YamlConfigTest, delimDot) {
    const std::string configPath = DUMMY_CONFIG_PATH;
    const char delim = '.';
    tbai::config::YamlConfig config(configPath, delim);

    ASSERT_EQ(config.get<std::string>("a.b"), "hello");
    ASSERT_EQ(config.get<int>("a.c"), 1);
    ASSERT_EQ(config.get<double>("a.d"), 3.14);
    ASSERT_EQ(config.get<int>("a.e.f"), 28);
}

TEST_F(YamlConfigTest, delimForwardslash) {
    const std::string configPath = DUMMY_CONFIG_PATH;
    const char delim = '/';
    tbai::config::YamlConfig config(configPath, delim);

    ASSERT_EQ(config.get<std::string>("a/b"), "hello");
    ASSERT_EQ(config.get<int>("a/c"), 1);
    ASSERT_EQ(config.get<double>("a/d"), 3.14);
    ASSERT_EQ(config.get<int>("a/e/f"), 28);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
