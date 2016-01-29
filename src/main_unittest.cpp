#include <gtest/gtest.h>
#include "core/config.hpp"
#include "core/logger.h"
#include <stdio.h>

// TODO: Move unittests and benchmarks, for example: splitter_unittest should be moved to util/splitter
// find src/ -name '*.cpp' -not -name '*_unittest*' -not -name '*_benchmark*'
// -o -name '*.cu' -not -name '*_unittest*' -not -name '*_benchmark*' | sort -k 1nr | cut -f2-

//--gtest_filter=AflEncoding_Compression_Inst/AflCompressionTest.CompressionOfRandomInts_size/0
//--gtest_repeat=10

using namespace std;
using namespace ddj;
namespace po = boost::program_options;

ConfigOptions GetProgramOptions()
{
	ConfigOptions options;

	// SET CONSOLE OPTIONS
	po::options_description consoleOptions("Config file options");
	options.ConsoleOptions.add(consoleOptions);

	// SET CONSOLE POSITIONAL OPTIONS
	po::positional_options_description consolePositionalOptions;
	options.ConsolePositionalOptions = consolePositionalOptions;

	// SET CONFIG FILE OPTIONS
	po::options_description configFileOptions("Config file options");
	configFileOptions.add_options()
	    ("TEST_DATA_LOG", po::value<std::string>()->default_value(""), "default file containing test time series data")
	    ("BENCHMARK_DATA_LOG", po::value<std::string>()->default_value(""), "default file containing benchmark time series data")
	    ("NYSE_DATA_1GB", po::value<std::string>()->default_value(""), "default file containing nyse time series data from openbook")
		("LOG_CONFIG", po::value<std::string>()->default_value("logger.prop", "file containing log4cplus configuration"))
	    ;
	options.ConfigFileOptions.add(configFileOptions);

	return options;
}

void initialize_logger()
{
  log4cplus::initialize();
  LogLog::getLogLog()->setInternalDebugging(true);
  auto loggerConfPath = ddj::Config::GetInstance()->GetValue<std::string>("LOG_CONFIG");
  PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT(loggerConfPath));
}

int main(int argc, char* argv[])
{
	ConfigDefinition configDef { argc, argv, "config.ini", GetProgramOptions() };
	ddj::Config::Initialize(configDef);
	initialize_logger();
	auto logger = Logger::getInstance("myDefaultLogger");
	LOG4CPLUS_DEBUG(logger, "START UNITTESTING");
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
