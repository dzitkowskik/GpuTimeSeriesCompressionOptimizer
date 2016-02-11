#include <benchmark/benchmark.h>
#include "core/config.hpp"
#include "core/logger.h"
#include <celero/Celero.h>

using namespace std;
using namespace ddj;
namespace po = boost::program_options;

const char** getBenchmarkFilter(char** argv)
{
    std::string filter = "--benchmark_filter=";
    const char** new_argv = new const char*[2];
    new_argv[0] = argv[0];
    new_argv[1] = filter.c_str();
    return new_argv;
}


void initialize_logger()
{
  log4cplus::initialize();
  LogLog::getLogLog()->setInternalDebugging(true);
  auto loggerConfPath = ddj::Config::GetInstance()->GetValue<std::string>("LOG_CONFIG");
  PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT(loggerConfPath));
}

int main(int argc, char** argv)
{
	initialize_logger();
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    celero::Run(argc, argv);
    return 0;
}
