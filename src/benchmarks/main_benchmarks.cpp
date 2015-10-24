#include <benchmark/benchmark.h>
#include "core/config.hpp"

const char** getBenchmarkFilter(char** argv)
{
    std::string filter = "--benchmark_filter=*";
    const char** new_argv = new const char*[2];
    new_argv[0] = argv[0];
    new_argv[1] = filter.c_str();
    return new_argv;
}

int main(int argc, char** argv)
{
    int new_argc = 2;
    const char** new_argv = getBenchmarkFilter(argv);

//    ddj::Config::GetInstance()->InitOptions(argc, argv, "benchmarks/benchmarks_config.ini");
    ::benchmark::Initialize(&new_argc, const_cast<const char**>(new_argv));

    // ::benchmark::Initialize(&argc, const_cast<const char**>(argv));
    ::benchmark::RunSpecifiedBenchmarks();

    delete [] new_argv;
    return 0;
}
