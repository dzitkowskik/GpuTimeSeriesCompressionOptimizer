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
#include "util/transform/cuda_array_transform.hpp"

namespace ddj
{

class CompressionOptimizerBenchmark : public BenchmarkBase {};

BENCHMARK_DEFINE_F(CompressionOptimizerBenchmark, BM_CompressionOptimizer_RandomInt)(benchmark::State& state)
{
    auto data = CastSharedCudaPtr<int, char>(GetIntRandomData(state.range_x(), 10,1000));

    while (state.KeepRunning())
	{
    	printf("start\n");
		CompressionOptimizer().OptimizeTree(data, DataType::d_int);
		printf("end\n");
	}

	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(CompressionOptimizerBenchmark, BM_CompressionOptimizer_RandomInt)->Arg(1<<20);


} /* namespace ddj */


