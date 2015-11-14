/*
 * compression_tree_benchmark_base.cpp
 *
 *  Created on: 11 lis 2015
 *      Author: ghash
 */

#include <benchmarks/compression_tree_benchmark_base.hpp>

namespace ddj
{

void CompressionTreeBenchmarkBase::Benchmark_Tree_Encoding(
		CompressionTree& tree,
		SharedCudaPtr<char> data,
		DataType type,
		benchmark::State& state)
{
	while (state.KeepRunning())
	{
		// ENCODE
		auto compr = tree.Compress(data);

		state.PauseTiming();
		compr.reset();
		state.ResumeTiming();
	}

	SetStatistics(state, type);
}

void CompressionTreeBenchmarkBase::Benchmark_Tree_Decoding(
		CompressionTree& tree,
		SharedCudaPtr<char> data,
		DataType type,
		benchmark::State& state)
{
	while (state.KeepRunning())
	{
		state.PauseTiming();
		auto compr = tree.Compress(data);
		state.ResumeTiming();

		// DECODE
		auto decompr = tree.Decompress(compr);

		state.PauseTiming();
		compr.reset();
		decompr.reset();
		state.ResumeTiming();
	}

	SetStatistics(state, type);
}

} /* namespace ddj */
