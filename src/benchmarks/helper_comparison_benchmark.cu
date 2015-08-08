#include "helpers/helper_comparison.cuh"
#include "helpers/helper_generator.hpp"
#include <benchmark/benchmark.h>

namespace ddj {

static void BM_HelperComparison_CompareDeviceArrays(benchmark::State& state)
{
    HelperGenerator generator;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        auto data_1 = generator.GenerateRandomFloatDeviceArray(state.range_x());
        auto data_2 = generator.GenerateRandomFloatDeviceArray(state.range_x());
        state.ResumeTiming();

        // COMPARE DATA
        CompareDeviceArrays(data_1->get(), data_2->get(), state.range_x());
    }
    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(float));
}
BENCHMARK(BM_HelperComparison_CompareDeviceArrays)->Arg(1<<20);

} /* namespace ddj */
