/*
 * cuda_array_copy_benchmark.cpp
 *
 *  Created on: Nov 20, 2015
 *      Author: Karol Dzitkowski
 */

#include "benchmarks/benchmark_base.hpp"
#include "cuda_array_copy.hpp"
#include "core/cuda_stream.hpp"
#include <celero/Celero.h>

namespace ddj
{

class CudaPtrBenchmark : public BenchmarkBase {};

BENCHMARK_DEFINE_F(CudaPtrBenchmark, BM_ConcatenateIntVectors_Sync)(benchmark::State& state)
{
    int N = 8;
    SharedCudaPtrVector<int> data;
    for(int i = 0; i < N; i++)
	   data.push_back( GetIntRandomData(state.range_x()) );

    printf("start\n");

    while (state.KeepRunning())
   	{
   		// CONCATENATE
   		auto concat = CudaArrayCopy().Concatenate(data);

   		state.PauseTiming();
   		concat.reset();
   		state.ResumeTiming();
   	}

   	SetStatistics(state, DataType::d_int);

   	printf("processed = %lu", state.items_processed());

   	printf("end\n");
}
BENCHMARK_REGISTER_F(CudaPtrBenchmark, BM_ConcatenateIntVectors_Sync)->Arg(1<<23);

BENCHMARK_DEFINE_F(CudaPtrBenchmark, BM_ConcatenateIntVectors_Parallel)(benchmark::State& state)
{
    int N = 8;
    SharedCudaPtrVector<int> data;
    for(int i = 0; i < N; i++)
	   data.push_back( GetIntRandomData(state.range_x()) );

    auto streams = CudaStream::make_shared(2);

    printf("start\n");

    while (state.KeepRunning())
   	{
   		// CONCATENATE
   		auto concat = CudaArrayCopy().ConcatenateParallel(data, streams);

   		state.PauseTiming();
   		concat.reset();
   		state.ResumeTiming();
   	}

   	SetStatistics(state, DataType::d_int);

   	printf("processed = %lu", state.items_processed());

   	printf("end\n");
}
BENCHMARK_REGISTER_F(CudaPtrBenchmark, BM_ConcatenateIntVectors_Parallel)->Arg(1<<23);

class ConcatCeleroFixture : public celero::TestFixture
{
	public:
		ConcatCeleroFixture() : _arraySize(1<<20) {}

		virtual std::vector<std::pair<int64_t, uint64_t>> getExperimentValues() const override
		{
			std::vector<std::pair<int64_t, uint64_t>> problemSpace;
			const int totalNumberOfTests = 1;
			for(int i = 0; i < totalNumberOfTests; i++)
				problemSpace.push_back(std::make_pair(int64_t(pow(2, i+22)), uint64_t(0)));
			return problemSpace;
		}

		virtual void setUp(int64_t experimentValue)
		{
			this->_arraySize = static_cast<int>(experimentValue);
			for(int i = 0; i < 20; i++)
				 _testData.push_back(
						 this->_generator.GenerateRandomIntDeviceArray(this->_arraySize, 10, 1000));
			this->_streams = CudaStream::make_shared(8);
		}

		virtual void tearDown()
		{
			this->_testData.clear();
			this->_streams.clear();
		}

		SharedCudaPtrVector<int> _testData;
		SharedCudaStreamVector _streams;
		size_t _arraySize;
		CudaArrayGenerator _generator;
};

BASELINE_F(CudaPtrBenchmark_Celero, Concatenate, ConcatCeleroFixture, 10, 10)
{
	CudaArrayCopy().Concatenate(this->_testData);
}

BENCHMARK_F(CudaPtrBenchmark_Celero, ConcatenateParallel, ConcatCeleroFixture, 10, 10)
{
	CudaArrayCopy().ConcatenateParallel(this->_testData, this->_streams);
}

} /* namespace ddj */



