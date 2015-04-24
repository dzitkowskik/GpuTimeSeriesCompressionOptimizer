#include "../helpers/helper_comparison.cuh"
#include "../helpers/helper_generator.h"
#include <benchmark/benchmark.h>

namespace ddj
{

static void BM_CompareDeviceArrays(benchmark::State& state)
{
    HelperGenerator generator;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        float* data_1 = generator.GenerateRandomFloatDeviceArray(state.range_x());
        float* data_2 = generator.GenerateRandomFloatDeviceArray(state.range_x());
        state.ResumeTiming();

        // COMPARE DATA
        CompareDeviceArrays(data_1, data_2, state.range_x());

        state.PauseTiming();
        cudaFree(data_1);
        cudaFree(data_2);
        state.ResumeTiming();
    }
    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(float));
}
BENCHMARK(BM_CompareDeviceArrays)->Arg(1<<20);

} /* namespace ddj */
