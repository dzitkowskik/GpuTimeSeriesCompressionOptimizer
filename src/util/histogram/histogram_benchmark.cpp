/*
 * histogram_benchmark.cpp
 *
 *  Created on: Feb 16, 2016
 *      Author: Karol Dzitkowski
 */

#include "histogram.hpp"
#include "benchmarks/benchmark_base.hpp"
#include <benchmark/benchmark.h>

namespace ddj
{

class HistogramBenchmark : public BenchmarkBase {};

BENCHMARK_DEFINE_F(HistogramBenchmark, BM_Histogram_PatternA_Int)(benchmark::State& state)
{
    auto data = GetFakeDataWithPatternA<int>(0, state.range_x());
    while (state.KeepRunning())
	{
    	Histogram().GetHistogram(data, 20);
	}
	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(HistogramBenchmark, BM_Histogram_PatternA_Int)
	->Arg(1<<20)->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(HistogramBenchmark, BM_GetDictionaryCounter_PatternA_Int)(benchmark::State& state)
{
    auto data = GetFakeDataWithPatternA<int>(0, state.range_x());
    while (state.KeepRunning())
	{
    	Histogram().GetDictionaryCounter(data);
	}
	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(HistogramBenchmark, BM_GetDictionaryCounter_PatternA_Int)
	->Arg(1<<20)->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

} /* namespace */
