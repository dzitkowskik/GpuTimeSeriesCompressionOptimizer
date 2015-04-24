#include "helpers/helper_generator.h"
#include "compression/delta/delta_encoding.h"
#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

namespace ddj
{

static void BM_Delta_Encode(benchmark::State& state)
{
    HelperGenerator generator;
    DeltaEncoding compression;
    DeltaEncodingMetadata<float> metadata;
    int out_size;

    while (state.KeepRunning())
    {
        state.PauseTiming();

        float* data = generator.GenerateRandomFloatDeviceArray(state.range_x());
        state.ResumeTiming();

        // ENCODE
        void* compr = compression.Encode(data, state.range_x(), out_size, metadata);

        state.PauseTiming();
        cudaFree(data);
        cudaFree(compr);
        state.ResumeTiming();
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
    DeltaEncodingMetadata<float> metadata;
    int out_size, out_size_decoded;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        float* data = generator.GenerateRandomFloatDeviceArray(state.range_x());
        void* compr = compression.Encode(data, state.range_x(), out_size, metadata);
        state.ResumeTiming();

        // DECODE
        float* decpr = compression.Decode<float>(compr, out_size, out_size_decoded, metadata);

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
BENCHMARK(BM_Delta_Decode)->Arg(1<<10)->Arg(1<<15)->Arg(1<<20);

} /* namespace ddj */
