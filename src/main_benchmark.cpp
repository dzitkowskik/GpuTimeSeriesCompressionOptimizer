#include <benchmark/benchmark.h>
#include "core/config.hpp"
#include <celero/Celero.h>

const char** getBenchmarkFilter(char** argv)
{
    std::string filter = "--benchmark_filter=";
    const char** new_argv = new const char*[2];
    new_argv[0] = argv[0];
    new_argv[1] = filter.c_str();
    return new_argv;
}

int main(int argc, char** argv)
{
	ddj::Config::GetInstance()->InitOptions(argc, argv, "config.ini");
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    celero::Run(argc, argv);
    return 0;
}
