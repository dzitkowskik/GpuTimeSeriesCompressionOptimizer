/*
 *  templates.hpp
 *
 *  Created on: 08-08-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_BENCHMARK_TEMPLATES_HPP_
#define DDJ_BENCHMARK_TEMPLATES_HPP_

#include "util/generator/cuda_array_generator.hpp"
#include <benchmark/benchmark.h>

namespace ddj {

template<class C>
static void Random_Float_Encode_Template(benchmark::State& state)
{
    CudaArrayGenerator generator;
    C compression;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        auto data = generator.GenerateRandomFloatDeviceArray(state.range_x());
        state.ResumeTiming();

        // ENCODE
        auto compr = compression.template Encode<float>(data);

        state.PauseTiming();
        data.reset();
        compr.clear();
        state.ResumeTiming();
    }

    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(float));
}

template<class C>
static void Random_Float_Decode_Template(benchmark::State& state)
{
	CudaArrayGenerator generator;
    C compression;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        auto data = generator.GenerateRandomFloatDeviceArray(state.range_x());
        auto compr = compression.template Encode<float>(data);
        state.ResumeTiming();

        // DECODE
        auto decpr = compression.template Decode<float>(compr);

        state.PauseTiming();
        data.reset();
        compr.clear();
        decpr.reset();
        state.ResumeTiming();
    }

    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(float));
}

template<class C>
static void Random_Int_Encode_Template(benchmark::State& state)
{
	CudaArrayGenerator generator;
    C compression;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        auto data = generator.GenerateRandomIntDeviceArray(state.range_x());
        state.ResumeTiming();

        // ENCODE
        auto compr = compression.template Encode<int>(data);

        state.PauseTiming();
        data.reset();
        compr.clear();
        state.ResumeTiming();
    }

    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(int));
}

template<class C>
static void Random_Int_Decode_Template(benchmark::State& state)
{
	CudaArrayGenerator generator;
    C compression;

    while (state.KeepRunning())
    {
        state.PauseTiming();
        auto data = generator.GenerateRandomIntDeviceArray(state.range_x());
        auto compr = compression.template Encode<int>(data);
        state.ResumeTiming();

        // DECODE
        auto decpr = compression.template Decode<int>(compr);

        state.PauseTiming();
        data.reset();
        compr.clear();
        decpr.reset();
        state.ResumeTiming();
    }

    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(int));
}


} /* namespace ddj */
#endif /* DDJ_BENCHMARK_TEMPLATES_HPP_ */
