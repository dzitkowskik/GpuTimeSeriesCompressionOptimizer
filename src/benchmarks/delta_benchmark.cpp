#include "helpers/helper_generator.hpp"
#include "compression/delta/delta_encoding.cuh"
#include "core/cuda_ptr.hpp"
#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

namespace ddj
{

template<class C, typename T>
static void EncodeTest(benchmark::State& state)
{
    HelperGenerator generator;
    C compression;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        auto data = generator.GenerateRandomFloatDeviceArray(state.range_x());
        state.ResumeTiming();

        // ENCODE
        auto compr = compression.Encode(data);
    }

    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(T));
}

static void BM_Delta_Float_Encode(benchmark::State& state)
{
    HelperGenerator generator;
    DeltaEncoding compression;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        auto data = generator.GenerateRandomFloatDeviceArray(state.range_x());
        state.ResumeTiming();

        // ENCODE
        auto compr = compression.Encode(data);

        state.PauseTiming();
        data.reset();
        compr.reset();
        state.ResumeTiming();
    }

    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(float));

}
BENCHMARK(BM_Delta_Float_Encode)->Arg(1<<15)->Arg(1<<20);

static void BM_Delta_Float_Decode(benchmark::State& state)
{
    HelperGenerator generator;
    DeltaEncoding compression;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        auto data = generator.GenerateRandomFloatDeviceArray(state.range_x());
        auto compr = compression.Encode(data);
        state.ResumeTiming();

        // DECODE
        auto decpr = compression.Decode<float>(compr);

        state.PauseTiming();
        data.reset();
        compr.reset();
        decpr.reset();
        state.ResumeTiming();
    }
    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(float));
}
BENCHMARK(BM_Delta_Float_Decode)->Arg(1<<15)->Arg(1<<20);

static void BM_Delta_Int_Encode(benchmark::State& state)
{
    HelperGenerator generator;
    DeltaEncoding compression;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        auto data = generator.GenerateRandomIntDeviceArray(state.range_x());
        state.ResumeTiming();

        // ENCODE
        auto compr = compression.Encode(data);

        state.PauseTiming();
        data.reset();
        compr.reset();
        state.ResumeTiming();
    }

    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(int));

}
BENCHMARK(BM_Delta_Int_Encode)->Arg(1<<15)->Arg(1<<20);

static void BM_Delta_Int_Decode(benchmark::State& state)
{
    HelperGenerator generator;
    DeltaEncoding compression;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        auto data = generator.GenerateRandomIntDeviceArray(state.range_x());
        auto compr = compression.Encode(data);
        state.ResumeTiming();

        // DECODE
        auto decpr = compression.Decode<int>(compr);
    }
    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(int));
}
BENCHMARK(BM_Delta_Int_Decode)->Arg(1<<15)->Arg(1<<20);

} /* namespace ddj */
