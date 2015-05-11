#include "helpers/helper_cudakernels.cuh"
#include <stdlib.h>
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <benchmark/benchmark.h>

namespace ddj {

static void BM_Modulo_Kernel_Simple(benchmark::State& state)
{
	srand (time(NULL));

	int n = state.range_x();
	int val = rand() % 101 + 1;
	int mod = rand() % 13 + 3;

    thrust::device_vector<int> d_data(n);
    int* d_data_raw = thrust::raw_pointer_cast(d_data.data());

    while (state.KeepRunning())
    {
        state.PauseTiming();
    	int val = rand() % 100 + 1;
    	int mod = rand() % 10 + 3;
		thrust::fill(d_data.begin(), d_data.end(), val);
        state.ResumeTiming();

        HelperCudaKernels::ModuloKernel(d_data_raw, n, mod);
    }
    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(int));
}
BENCHMARK(BM_Modulo_Kernel_Simple)->Arg(1<<10)->Arg(1<<16)->Arg(1<<20);

static void BM_Modulo_Thrust_Simple(benchmark::State& state)
{
	srand (time(NULL));

	int n = state.range_x();
	int val = rand() % 101 + 1;
	int mod = rand() % 13 + 3;

    thrust::device_vector<int> d_data(n);
    int* d_data_raw = thrust::raw_pointer_cast(d_data.data());

    while (state.KeepRunning())
    {
        state.PauseTiming();
    	int val = rand() % 100 + 1;
    	int mod = rand() % 10 + 3;
		thrust::fill(d_data.begin(), d_data.end(), val);
        state.ResumeTiming();

        HelperCudaKernels::ModuloThrust(d_data_raw, n, mod);
    }
    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(int));
}
BENCHMARK(BM_Modulo_Thrust_Simple)->Arg(1<<10)->Arg(1<<15)->Arg(1<<20);

} /* namespace ddj */
