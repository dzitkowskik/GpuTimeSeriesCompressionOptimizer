#include "helpers/helper_generator.hpp"
#include "compression/scale/scale_encoding.cuh"
#include "core/cuda_ptr.hpp"
#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

namespace ddj {

static void BM_Scale_Float_Encode(benchmark::State& state)
{
    HelperGenerator generator;
    ScaleEncoding compression;

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
BENCHMARK(BM_Scale_Float_Encode)->Arg(1<<10)->Arg(1<<15)->Arg(1<<20);

static void BM_Scale_Float_Decode(benchmark::State& state)
{
    HelperGenerator generator;
    ScaleEncoding compression;

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
BENCHMARK(BM_Scale_Float_Decode)->Arg(1<<10)->Arg(1<<15)->Arg(1<<20);

} /* namespace ddj */
