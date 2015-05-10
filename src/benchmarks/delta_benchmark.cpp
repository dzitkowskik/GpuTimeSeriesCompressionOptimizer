#include "helpers/helper_generator.h"
#include "compression/delta/delta_encoding.cuh"
#include "core/cuda_ptr.h"
#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

namespace ddj
{

static void BM_Delta_Encode(benchmark::State& state)
{
    HelperGenerator generator;
    DeltaEncoding compression;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        auto data = CudaPtr<float>::make_shared(
        	generator.GenerateRandomFloatDeviceArray(state.range_x()), state.range_x());
        state.ResumeTiming();

        // ENCODE
        auto compr = compression.Encode(data);
    }

    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(float));
}
BENCHMARK(BM_Delta_Encode)->Arg(1<<10)->Arg(1<<15)->Arg(1<<20);

static void BM_Delta_Decode(benchmark::State& state)
{
    HelperGenerator generator;
    DeltaEncoding compression;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        auto data = CudaPtr<float>::make_shared(
        	generator.GenerateRandomFloatDeviceArray(state.range_x()), state.range_x());
        auto compr = compression.Encode(data);
        state.ResumeTiming();

        // DECODE
        auto decpr = compression.Decode<float>(compr);
    }
    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(float));
}
BENCHMARK(BM_Delta_Decode)->Arg(1<<10)->Arg(1<<15)->Arg(1<<20);

} /* namespace ddj */
