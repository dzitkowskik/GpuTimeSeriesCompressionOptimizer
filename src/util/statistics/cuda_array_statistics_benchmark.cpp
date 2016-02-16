/*
 * cuda_array_statistics_benchmark.cpp
 *
 *  Created on: Feb 16, 2016
 *      Author: Karol Dzitkowski
 */

#include "cuda_array_statistics.hpp"
#include "benchmarks/benchmark_base.hpp"
#include <benchmark/benchmark.h>

namespace ddj
{

class CudaArrayStatisticsBenchmark : public BenchmarkBase {};

BENCHMARK_DEFINE_F(CudaArrayStatisticsBenchmark, BM_Mean_PatternA_Int)(benchmark::State& state)
{
    auto data = GetFakeDataWithPatternA<int>(0, state.range_x());
    while (state.KeepRunning())
	{
    	CudaArrayStatistics().Mean(data);
	}
	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CudaArrayStatisticsBenchmark, BM_Mean_PatternA_Int)
	->Arg(1<<20)->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(CudaArrayStatisticsBenchmark, BM_MinBitCnt_PatternA_Int)(benchmark::State& state)
{
    auto data = GetFakeDataWithPatternA<int>(0, state.range_x());
    while (state.KeepRunning())
	{
    	CudaArrayStatistics().MinBitCnt(data);
	}
	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CudaArrayStatisticsBenchmark, BM_MinBitCnt_PatternA_Int)
	->Arg(1<<20)->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(CudaArrayStatisticsBenchmark, BM_MinMax_PatternA_Int)(benchmark::State& state)
{
    auto data = GetFakeDataWithPatternA<int>(0, state.range_x());
    while (state.KeepRunning())
	{
    	CudaArrayStatistics().MinMax(data);
	}
	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CudaArrayStatisticsBenchmark, BM_MinMax_PatternA_Int)
	->Arg(1<<20)->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(CudaArrayStatisticsBenchmark, BM_Precision_PatternA_Int)(benchmark::State& state)
{
    auto data = GetFakeDataWithPatternA<int>(0, state.range_x());
    while (state.KeepRunning())
	{
    	CudaArrayStatistics().Precision(data);
	}
	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CudaArrayStatisticsBenchmark, BM_Precision_PatternA_Int)
	->Arg(1<<20)->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(CudaArrayStatisticsBenchmark, BM_RlMetric_PatternA_Int)(benchmark::State& state)
{
    auto data = GetFakeDataWithPatternA<int>(0, state.range_x());
    while (state.KeepRunning())
	{
    	CudaArrayStatistics().RlMetric(data);
	}
	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CudaArrayStatisticsBenchmark, BM_RlMetric_PatternA_Int)
	->Arg(1<<20)->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(CudaArrayStatisticsBenchmark, BM_Sorted_PatternA_Int)(benchmark::State& state)
{
    auto data = GetFakeDataWithPatternA<int>(0, state.range_x());
    while (state.KeepRunning())
	{
    	CudaArrayStatistics().Sorted(data);
	}
	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CudaArrayStatisticsBenchmark, BM_Sorted_PatternA_Int)
	->Arg(1<<20)->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(CudaArrayStatisticsBenchmark, BM_GenerateStatistics_PatternA_Int)(benchmark::State& state)
{
    auto data = CastSharedCudaPtr<int, char>(GetFakeDataWithPatternA<int>(0, state.range_x()));
    while (state.KeepRunning())
	{
    	CudaArrayStatistics().GenerateStatistics(data, DataType::d_int);
	}
	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CudaArrayStatisticsBenchmark, BM_GenerateStatistics_PatternA_Int)
	->Arg(1<<20)->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

} /* namespace ddj */



