#include "helpers/helper_generator.hpp"
#include "core/cuda_ptr.hpp"
#include "compression/patch/patch.cuh"
#include "core/operators.cuh"
#include "helpers/helper_print.hpp"
#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

namespace ddj
{

static void BM_Patch_ConsecutiveNumbers_Low_High(benchmark::State& state)
{
    int n = state.range_x();
    HelperGenerator generator;
    auto d_data = generator.GenerateConsecutiveIntDeviceArray(n);
    OutsideOperator<int> op{n/3, 2*n/3};

    while (state.KeepRunning())
    {
        state.PauseTiming();
        PatchedData<int, OutsideOperator<int>> patch(op);
        state.ResumeTiming();

        patch.Init(d_data);
    }

    long long int it_processed = state.iterations() * state.range_x();
    state.SetItemsProcessed(it_processed);
    state.SetBytesProcessed(it_processed * sizeof(int));

}
BENCHMARK(BM_Patch_ConsecutiveNumbers_Low_High)->Arg(1<<15)->Arg(1<<20);



} /* namespace ddj */
