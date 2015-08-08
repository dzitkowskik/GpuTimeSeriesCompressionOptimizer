#include "helpers/helper_generator.hpp"
#include "core/cuda_ptr.hpp"
#include "benchmark_base_inline.hpp"
#include "util/histogram/basic_thrust_histogram.hpp"
#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

namespace ddj
{

static void BM_HISTOGRAM_BasicThrustHistogram_ConsecutiveNumbers(benchmark::State& state)
{
    HelperGenerator generator;
    BasicThrustHistogram histogram;
    auto d_data = generator.GenerateConsecutiveIntDeviceArray(state.range_x());

    while (state.KeepRunning())
    {
        histogram.IntegerHistogram(d_data);
    }

    Set_Statistics(state);
}
BENCHMARK(BM_HISTOGRAM_BasicThrustHistogram_ConsecutiveNumbers)->Arg(1<<10)->Arg(1<<15)->Arg(1<<20);

static void BM_HISTOGRAM_BasicThrustHistogram_RandomNumbers(benchmark::State& state)
{
    HelperGenerator generator;
    BasicThrustHistogram histogram;

    while (state.KeepRunning())
    {
        auto d_data = PrepareRandomIntData(state);
        histogram.IntegerHistogram(d_data);
        ManualReleaseRandomIntData(state, d_data);
    }

    Set_Statistics(state);
}
BENCHMARK(BM_HISTOGRAM_BasicThrustHistogram_RandomNumbers)->Arg(1<<10)->Arg(1<<15)->Arg(1<<20);

} /* namespace ddj */
