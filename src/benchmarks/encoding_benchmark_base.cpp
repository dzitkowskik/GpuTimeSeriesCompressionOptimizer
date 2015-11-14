/*
 *  encoding_banchmark_base.cpp
 *
 *  Created on: 25/10/2015
 *      Author: Karol Dzitkowski
 */

#include "benchmarks/encoding_benchmark_base.hpp"

namespace ddj {

void EncodingBenchmarkBase::Benchmark_Encoding(
	Encoding& encoding,
	SharedCudaPtr<char> data,
	DataType type,
	benchmark::State& state)
{
	while (state.KeepRunning())
	{
		state.PauseTiming();
		SharedCudaPtr<char> data_copy = data->copy();
		state.ResumeTiming();

		// ENCODE
		auto compr = encoding.Encode(data_copy, type);

		state.PauseTiming();
		compr.clear();
		state.ResumeTiming();
	}

	SetStatistics(state, type);
}

void EncodingBenchmarkBase::Benchmark_Decoding(
	Encoding& encoding,
	SharedCudaPtr<char> data,
	DataType type,
	benchmark::State& state)
{
	while (state.KeepRunning())
	{
		state.PauseTiming();
		SharedCudaPtr<char> data_copy = data->copy();
		auto compr = encoding.Encode(data_copy, type);
		state.ResumeTiming();

		// DECODE
		auto decompr = encoding.Decode(compr, type);

		state.PauseTiming();
		compr.clear();
		decompr.reset();
		state.ResumeTiming();
	}

	SetStatistics(state, type);
}

} /* namespace ddj */
