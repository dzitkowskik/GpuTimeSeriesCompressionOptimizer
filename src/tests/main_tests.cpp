#include <gtest/gtest.h>
#include "core/config.hpp"
#include <stdio.h>

int main(int argc, char* argv[])
{
    ddj::Config::GetInstance()->InitOptions(argc, argv, "config.ini");
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::FLAGS_gtest_repeat = 1;
    return RUN_ALL_TESTS();
}
