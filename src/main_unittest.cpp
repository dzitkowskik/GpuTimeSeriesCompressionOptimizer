#include <gtest/gtest.h>
#include "core/config.hpp"
#include "core/logger.h"
#include <stdio.h>

// TODO: Move unittests and benchmarks, for example: splitter_unittest should be moved to util/splitter
// find src/ -name '*.cpp' -not -name '*_unittest*' -not -name '*_benchmark*'
// -o -name '*.cu' -not -name '*_unittest*' -not -name '*_benchmark*' | sort -k 1nr | cut -f2-

//--gtest_filter=AflEncoding_Compression_Inst/AflCompressionTest.CompressionOfRandomInts_size/0
//--gtest_repeat=10

void initialize_logger()
{
  log4cplus::initialize();
  LogLog::getLogLog()->setInternalDebugging(true);
  PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT("logger.prop"));
}

int main(int argc, char* argv[])
{
    ddj::Config::GetInstance()->InitOptions(argc, argv, "config.ini");
    ::testing::InitGoogleTest(&argc, argv);
//    ::testing::FLAGS_gtest_repeat = 1;

    auto value = ddj::Config::GetInstance()->GetValue<std::string>("TEST_DATA_LOG");
    printf("%s\n", value.c_str());

    return RUN_ALL_TESTS();
}
