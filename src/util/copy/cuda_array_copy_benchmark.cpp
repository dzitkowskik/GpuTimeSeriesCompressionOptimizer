/*
 * cuda_array_copy_benchmark.cpp
 *
 *  Created on: Nov 20, 2015
 *      Author: Karol Dzitkowski
 */

#include "benchmarks/benchmark_base.hpp"
#include "cuda_array_copy.hpp"

namespace ddj
{

class CudaPtrBenchmark : public BenchmarkBase {};

BENCHMARK_DEFINE_F(CudaPtrBenchmark, BM_ConcatenateIntVectors_Parallel)(benchmark::State& state)
{
    int N = 8;
    SharedCudaPtrVector<int> data;
    for(int i = 0; i < N; i++)
	   data.push_back( GetIntRandomData(state.range_x()) );

    auto streams = CudaStream::make_shared(1);
    cudaDeviceSynchronize();

    printf("start\n");

    while (state.KeepRunning())
   	{
   		// CONCATENATE
   		auto concat = CudaArrayCopy().ConcatenateParallel(data, streams);

   		state.PauseTiming();
   		concat.reset();
   		state.ResumeTiming();
   	}
    cudaDeviceSynchronize();
   	SetStatistics(state, DataType::d_int);

   	printf("processed = %d", state.items_processed());

   	printf("end\n");
}
BENCHMARK_REGISTER_F(CudaPtrBenchmark, BM_ConcatenateIntVectors_Parallel)->Arg(1<<22);

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
BENCHMARK_REGISTER_F(CudaPtrBenchmark, BM_ConcatenateIntVectors_Sync)->Arg(1<<22);


} /* namespace ddj */



