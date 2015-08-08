#include "helpers/helper_cudakernels.cuh"
#include "core/cuda_ptr.hpp"
#include "benchmark_base_inline.hpp"

#include <stdlib.h>
#include <time.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <benchmark/benchmark.h>

namespace ddj {

static void BM_Modulo_Kernel_Normal(benchmark::State& state)
{
    HelperCudaKernels kernels;
    int mod = 123;

    while (state.KeepRunning())
    {
        auto d_data_shared = PrepareRandomIntData(state);
        kernels.ModuloKernel(d_data_shared, mod);
        ManualReleaseRandomIntData(state, d_data_shared);
    }

    Set_Statistics(state);
}
BENCHMARK(BM_Modulo_Kernel_Normal)->Arg(1<<20);

static void BM_Modulo_Kernel_InPlace(benchmark::State& state)
{
    HelperCudaKernels kernels;
    int mod = 123;

    while (state.KeepRunning())
    {
        auto d_data_shared = PrepareRandomIntData(state);
        kernels.ModuloInPlaceKernel(d_data_shared, mod);
        ManualReleaseRandomIntData(state, d_data_shared);
    }

    Set_Statistics(state);
}
BENCHMARK(BM_Modulo_Kernel_InPlace)->Arg(1<<20);

} /* namespace ddj */
