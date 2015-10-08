/*
 * benchmark_base_inline.hpp 04-07-2015 Karol Dzitkowski
 */

#ifndef DDJ_BENCHMARK_BASE_INLINE_HPP_
#define DDJ_BENCHMARK_BASE_INLINE_HPP_

#include "util/generator/cuda_array_generator.hpp"
#include <benchmark/benchmark.h>

namespace ddj {

inline void Set_Statistics(benchmark::State& state)
{
    auto it_processed = static_cast<int64_t>(state.iterations() * state.range_x());
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(int));
}

inline SharedCudaPtr<int> PrepareRandomIntData(benchmark::State& state)
{
    state.PauseTiming();
    CudaArrayGenerator generator;
    auto d_data_shared = generator.GenerateRandomIntDeviceArray(state.range_x());
    state.ResumeTiming();
    return d_data_shared;
}

inline void ManualReleaseRandomIntData(benchmark::State& state, SharedCudaPtr<int> data)
{
    state.PauseTiming();
    data.reset();
    state.ResumeTiming();
}

} /* namespace ddj */
#endif /* DDJ_BENCHMARK_BASE_INLINE_HPP_ */
