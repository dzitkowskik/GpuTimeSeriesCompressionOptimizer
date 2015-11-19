#include "benchmarks/benchmark_base.hpp"

namespace ddj
{

class CudaPtrBenchmark : public BenchmarkBase {};

BENCHMARK_DEFINE_F(CudaPtrBenchmark, BM_ConcatenateIntVectors)(benchmark::State& state)
{
    int N = 10;
    SharedCudaPtrVector<int> data;
    for(int i = 0; i < N; i++)
	   data.push_back( GetIntRandomData(state.range_x()) );

    while (state.KeepRunning())
   	{
   		// CONCATENATE
   		auto concat = Concatenate(data);

   		state.PauseTiming();
   		concat.reset();
   		state.ResumeTiming();
   	}

   	SetStatistics(state, type);
}
BENCHMARK_REGISTER_F(CudaPtrBenchmark, BM_ConcatenateIntVectors)->Arg(1<<15)->Arg(1<<20);


} /* namespace ddj */
