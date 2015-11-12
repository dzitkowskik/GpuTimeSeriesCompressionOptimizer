/*
 *  encoding_banchmark_base.hpp
 *
 *  Created on: 25/10/2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_COMPRESSION_BENCHMARK_BASE_HPP_
#define DDJ_COMPRESSION_BENCHMARK_BASE_HPP_

#include <benchmark/benchmark.h>
#include "core/cuda_ptr.hpp"
#include "compression/data_type.hpp"
#include "compression/encoding.hpp"
#include "benchmarks/benchmark_base.hpp"

namespace ddj {

class EncodingBenchmarkBase : public BenchmarkBase
{
public:
	void Benchmark_Encoding(
			Encoding& encoding,
			SharedCudaPtr<char> data,
			DataType type,
			benchmark::State& state);

	void Benchmark_Decoding(
			Encoding& encoding,
			SharedCudaPtr<char> data,
			DataType type,
			benchmark::State& state);

private:
	void SetStatistics(benchmark::State& state, DataType type);
};


} /* namespace ddj */
#endif /* DDJ_COMPRESSION_BENCHMARK_BASE_HPP_ */
