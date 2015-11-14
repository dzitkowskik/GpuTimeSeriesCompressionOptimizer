/*
 *  compression_tree_benchmark_base.hpp
 *
 *  Created on: 11/11/2015
 *      Author: Karol Dzitkowski
 */

#ifndef COMPRESSION_TREE_BENCHMARK_BASE_HPP_
#define COMPRESSION_TREE_BENCHMARK_BASE_HPP_

#include "benchmarks/benchmark_base.hpp"
#include "tree/compression_tree.hpp"
#include "compression/data_type.hpp"
#include "helpers/helper_device.hpp"

namespace ddj {

class CompressionTreeBenchmarkBase : public BenchmarkBase
{
public:
	CompressionTreeBenchmarkBase()
	{
		HelperDevice hc;
		int devId = hc.SetCudaDeviceWithMaxFreeMem();
		printf("TEST SET UP ON DEVICE %d\n", devId);
	}
	~CompressionTreeBenchmarkBase(){}

public:
	void Benchmark_Tree_Encoding(
			CompressionTree& tree,
			SharedCudaPtr<char> data,
			DataType type,
			benchmark::State& state);

	void Benchmark_Tree_Decoding(
			CompressionTree& tree,
			SharedCudaPtr<char> data,
			DataType type,
			benchmark::State& state);
};

} /* namespace ddj */
#endif /* COMPRESSION_TREE_BENCHMARK_BASE_HPP_ */
