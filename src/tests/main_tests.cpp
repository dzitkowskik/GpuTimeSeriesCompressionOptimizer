#include <gtest/gtest.h>
#include "core/config.hpp"
#include <stdio.h>

// TODO: Move unittests and benchmarks, for example: splitter_unittest should be moved to util/splitter
// find src/ -name '*.cpp' -not -name '*_unittest*' -not -name '*_benchmark*'
// -o -name '*.cu' -not -name '*_unittest*' -not -name '*_benchmark*' | sort -k 1nr | cut -f2-

int main(int argc, char* argv[])
{
    ddj::Config::GetInstance()->InitOptions(argc, argv, "config.ini");
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::FLAGS_gtest_repeat = 1;
    return RUN_ALL_TESTS();
}
