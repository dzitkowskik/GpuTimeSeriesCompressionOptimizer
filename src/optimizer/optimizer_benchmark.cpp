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

BENCHMARK_DEFINE_F(CompressionOptimizerBenchmark, BM_CompressionOptimizer_Phase1_RandomInt_CompressByBestTree)(benchmark::State& state)
{
	auto randomIntData = GetIntRandomData(state.range_x(), 10,1000);
    auto data = CastSharedCudaPtr<int, char>(randomIntData);
	auto trainSize = state.range_x()/100;
    auto trainData = CudaPtr<int>::make_shared();
    trainData->fill(randomIntData->get(), trainSize);
    auto trainDataChar = CastSharedCudaPtr<int, char>(trainData);
	auto results = PathGenerator().Phase1(trainDataChar, EncodingType::none, DataType::d_int, Statistics(), 0);
	std::sort(results.begin(), results.end(), [&](PossibleTree A, PossibleTree B){ return A.second < B.second; });
	results[0].first.Fix();
//	results[0].first.Print(results[0].second);

    while (state.KeepRunning())
	{
		auto compressed = results[0].first.Compress(data);
	}

	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CompressionOptimizerBenchmark, BM_CompressionOptimizer_Phase1_RandomInt_CompressByBestTree)->Arg(1<<20);


} /* namespace ddj */


