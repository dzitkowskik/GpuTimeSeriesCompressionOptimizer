#include "helpers/helper_generator.hpp"
#include "compression/rle/thrust_rle.cuh"
#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

namespace ddj {

static void BM_Thrust_RLE_Encode(benchmark::State& state)
{
    HelperGenerator generator;
    ThrustRleCompression compression;
    int out_size;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        float* data = generator.GenerateRandomFloatDeviceArray(state.range_x());
        state.ResumeTiming();

        // ENCODE
        void* compr = compression.Encode(data, state.range_x(), out_size);

        state.PauseTiming();
        cudaFree(data);
        cudaFree(compr);
        state.ResumeTiming();
    }
    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(float));
}
BENCHMARK(BM_Thrust_RLE_Encode)->Arg(1<<10)->Arg(1<<15)->Arg(1<<20);

static void BM_Thrust_RLE_Decode(benchmark::State& state)
{
    HelperGenerator generator;
    ThrustRleCompression compression;
    int out_size, out_size_decoded;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        float* data = generator.GenerateRandomFloatDeviceArray(state.range_x());
        void* compr = compression.Encode(data, state.range_x(), out_size);
        state.ResumeTiming();

        // DECODE
        float* decpr = compression.Decode<float>(compr, out_size, out_size_decoded);

        state.PauseTiming();
        cudaFree(data);
        cudaFree(compr);
        cudaFree(decpr);
        state.ResumeTiming();
    }
    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(float));
}
BENCHMARK(BM_Thrust_RLE_Decode)->Arg(1<<10)->Arg(1<<15)->Arg(1<<20);

} /* namespace ddj */
