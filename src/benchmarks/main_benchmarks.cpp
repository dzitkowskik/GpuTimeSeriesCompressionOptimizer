#include <benchmark/benchmark.h>
#include "../logger.h"
#include "../config.h"

static void BM_StringCreation(benchmark::State& state) {
  while (state.KeepRunning())
    std::string empty_string;
}
// Register the function as a benchmark
BENCHMARK(BM_StringCreation);

// Define another benchmark
static void BM_StringCopy(benchmark::State& state) {
  std::string x = "hello";
  while (state.KeepRunning())
    std::string copy(x);
}
BENCHMARK(BM_StringCopy);

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
