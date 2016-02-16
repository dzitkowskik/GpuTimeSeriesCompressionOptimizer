/*
 *  optimizer_benchmark.cpp
 *
 *  Created on: 14/11/2015
 *      Author: Karol Dzitkowski
 */

#include "benchmarks/compression_tree_benchmark_base.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "compression/default_encoding_factory.hpp"
#include "optimizer/compression_optimizer.hpp"
#include "optimizer/path_generator.hpp"
#include "util/transform/cuda_array_transform.hpp"
#include <benchmark/benchmark.h>

namespace ddj
{

// --benchmark_filter=CompressionOptimizerBenchmark/*
class CompressionOptimizerBenchmark : public BenchmarkBase {};

BENCHMARK_DEFINE_F(CompressionOptimizerBenchmark, BM_CompressionOptimizer_RandomInt)(benchmark::State& state)
{
    auto data = CastSharedCudaPtr<int, char>(GetIntRandomData(state.range_x(), 10,1000));
    while (state.KeepRunning())
	{
		CompressionOptimizer().OptimizeTree(data, DataType::d_int);
	}

	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CompressionOptimizerBenchmark, BM_CompressionOptimizer_RandomInt)->Arg(1<<20);

BENCHMARK_DEFINE_F(CompressionOptimizerBenchmark, BM_CompressByBestTree_ChoosenInPhase1_RandomInt)(benchmark::State& state)
{
	auto randomIntData = GetIntRandomData(state.range_x(), 10,1000);
    auto data = CastSharedCudaPtr<int, char>(randomIntData);
	auto trainSize = state.range_x()/100;
    auto trainData = CudaPtr<int>::make_shared();
    trainData->fill(randomIntData->get(), trainSize);
    auto trainDataChar = CastSharedCudaPtr<int, char>(trainData);

	auto results = CompressionOptimizer().FullStatisticsUpdate(
			trainDataChar,
			EncodingType::none,
			DataType::d_int,
			0);

	std::sort(results.begin(), results.end(), [&](PossibleTree A, PossibleTree B){ return A.second < B.second; });
	results[0].first.Fix();
	results[0].first.Print();

    while (state.KeepRunning())
	{
		auto compressed = results[0].first.Compress(data);
	}

	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CompressionOptimizerBenchmark, BM_CompressByBestTree_ChoosenInPhase1_RandomInt)->Arg(1<<20);

BENCHMARK_DEFINE_F(CompressionOptimizerBenchmark, BM_FullStatisticsUpdate_RawPhase1_RandomInt)(benchmark::State& state)
{
	auto randomIntData = GetIntRandomData(state.range_x(), 10,1000);
    auto data = CastSharedCudaPtr<int, char>(randomIntData);

    while (state.KeepRunning())
	{
		auto results = CompressionOptimizer().FullStatisticsUpdate(
				data,
				EncodingType::none,
				DataType::d_int,
				0);
	}

	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CompressionOptimizerBenchmark, BM_FullStatisticsUpdate_RawPhase1_RandomInt)
->Arg(1<<16)->Arg(1<<17)->Arg(1<<18)->Arg(1<<19)->Arg(1<<20);

BENCHMARK_DEFINE_F(CompressionOptimizerBenchmark, BM_FullStatisticsUpdate_RawPhase1_Time)(benchmark::State& state)
{
	auto intData = CudaArrayTransform().Cast<time_t, int>(GetTsIntDataFromFile(state.range_x()));
    auto data = CastSharedCudaPtr<int, char>(intData);

    while (state.KeepRunning())
	{
		auto results = CompressionOptimizer().FullStatisticsUpdate(
				data,
				EncodingType::none,
				DataType::d_int,
				0);
	}

	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CompressionOptimizerBenchmark, BM_FullStatisticsUpdate_RawPhase1_Time)->Arg(1<<20);

BENCHMARK_DEFINE_F(CompressionOptimizerBenchmark, BM_UpdateStatistics_RawPhase2_RandomInt)(benchmark::State& state)
{
	CompressionOptimizer optimizer;

	auto randomIntData = GetIntRandomData(state.range_x(), 10,1000);
	auto data = CastSharedCudaPtr<int, char>(randomIntData);
	auto trainSize = state.range_x()/100;
	auto trainData = CudaPtr<int>::make_shared();
	trainData->fill(randomIntData->get(), trainSize);
	auto trainDataChar = CastSharedCudaPtr<int, char>(trainData);

	auto results = optimizer.FullStatisticsUpdate(
			trainDataChar,
			EncodingType::none,
			DataType::d_int,
			0);

	std::sort(results.begin(), results.end(), [&](PossibleTree A, PossibleTree B){ return A.second < B.second; });
	results[0].first.Fix();

	while (state.KeepRunning())
	{
		results[0].first.UpdateStatistics(optimizer.GetStatistics());
	}

	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CompressionOptimizerBenchmark, BM_UpdateStatistics_RawPhase2_RandomInt)->Arg(1<<20);

BENCHMARK_DEFINE_F(CompressionOptimizerBenchmark, BM_TryCorrectTree_RawPhase3_RandomInt)(benchmark::State& state)
{
	CompressionOptimizer optimizer;

	auto randomIntData = GetIntRandomData(state.range_x(), 10,1000);
	auto data = CastSharedCudaPtr<int, char>(randomIntData);
	auto trainSize = state.range_x()/100;
	auto trainData = CudaPtr<int>::make_shared();
	trainData->fill(randomIntData->get(), trainSize);
	auto trainDataChar = CastSharedCudaPtr<int, char>(trainData);

	auto results = optimizer.FullStatisticsUpdate(
			trainDataChar,
			EncodingType::none,
			DataType::d_int,
			0);

	std::sort(results.begin(), results.end(), [&](PossibleTree A, PossibleTree B){ return A.second < B.second; });
	results[0].first.Fix();
	results[0].first.UpdateStatistics(optimizer.GetStatistics());
	results[0].first.SetStatistics(optimizer.GetStatistics());
	auto optimalTree = OptimalTree::make_shared(results[0].first);

	while (state.KeepRunning())
	{
		optimalTree->GetTree().FindNode(0)->SetCompressionRatio(1.0);
		bool corrected = optimalTree->TryCorrectTree();
	}
	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CompressionOptimizerBenchmark, BM_TryCorrectTree_RawPhase3_RandomInt)->Arg(1<<20);

BENCHMARK_DEFINE_F(CompressionOptimizerBenchmark, BM_CompressData_RandomInt)(benchmark::State& state)
{
	CompressionOptimizer optimizer;

	auto randomIntData = GetIntRandomData(state.range_x(), 10,1000);
	auto data = CastSharedCudaPtr<int, char>(randomIntData);

	while (state.KeepRunning())
	{
		optimizer.CompressData(data, DataType::d_int);
	}
	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CompressionOptimizerBenchmark, BM_CompressData_RandomInt)->Range(1<<16, 1<<25);

BENCHMARK_DEFINE_F(CompressionOptimizerBenchmark, BM_CompressData_PatternA_Int)(benchmark::State& state)
{
	CompressionOptimizer optimizer;

	auto data = CastSharedCudaPtr<int, char>(
			GetFakeDataWithPatternA<int>(0, state.range_x()));

	while (state.KeepRunning())
	{
		optimizer.CompressData(data, DataType::d_int);
	}
	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CompressionOptimizerBenchmark, BM_CompressData_PatternA_Int)->Range(1<<16, 1<<25);

BENCHMARK_DEFINE_F(CompressionOptimizerBenchmark, BM_CompressData_PatternB_Int)(benchmark::State& state)
{
	CompressionOptimizer optimizer;

	auto data = CastSharedCudaPtr<int, char>(
			GetFakeDataWithPatternA<int>(0, state.range_x()));

	while (state.KeepRunning())
	{
		optimizer.CompressData(data, DataType::d_int);
	}
	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CompressionOptimizerBenchmark, BM_CompressData_PatternB_Int)->Range(1<<16, 1<<25);

BENCHMARK_DEFINE_F(CompressionOptimizerBenchmark, BM_CompressData_Time)(benchmark::State& state)
{
	CompressionOptimizer optimizer;

	auto randomIntData = GetFakeDataForTime(state.range_x());
	auto data = CastSharedCudaPtr<time_t, char>(randomIntData);

	while (state.KeepRunning())
	{
		optimizer.CompressData(data, DataType::d_time);
	}
	SetStatistics(state, DataType::d_time);
}
BENCHMARK_REGISTER_F(CompressionOptimizerBenchmark, BM_CompressData_Time)->Range(1<<16, 1<<23);

} /* namespace ddj */


