#include <benchmark/benchmark.h>
#include "../core/logger.h"
#include "../core/config.h"

void initialize_logger()
{
  log4cplus::initialize();
  LogLog::getLogLog()->setInternalDebugging(true);
  PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT("logger.prop"));
}

int main(int argc, char** argv)
{
    ddj::Config::GetInstance()->InitOptions(argc, argv, "config.ini");
    initialize_logger();
    ::benchmark::Initialize(&argc, const_cast<const char**>(argv));
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
