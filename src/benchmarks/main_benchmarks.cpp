#include <benchmark/benchmark.h>
#include "../core/config.h"

int main(int argc, char** argv)
{
    ddj::Config::GetInstance()->InitOptions(argc, argv, "config.ini");
    ::benchmark::Initialize(&argc, const_cast<const char**>(argv));
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
