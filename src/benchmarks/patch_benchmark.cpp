#include "util/generator/cuda_array_generator.hpp"
#include "compression/patch/patch_encoding.hpp"
#include "core/operators.cuh"
#include "templates.hpp"

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

namespace ddj
{

static void BM_Patch_ConsecutiveNumbers_OutsideOperator_Int_Encode(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int, char>(
    		CudaArrayGenerator().GenerateConsecutiveIntDeviceArray(n));
    OutsideOperator<int> op{n/3, 2*n/3};
    PatchEncoding<OutsideOperator<int>> encoding(op);
    Benchmark_Encoding(encoding, data, DataType::d_int, state);
}
BENCHMARK(BM_Patch_ConsecutiveNumbers_OutsideOperator_Int_Encode)->Arg(1<<15)->Arg(1<<20);

static void BM_Patch_ConsecutiveNumbers_OutsideOperator_Int_Decode(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int, char>(
    		CudaArrayGenerator().GenerateConsecutiveIntDeviceArray(n));
    OutsideOperator<int> op{n/3, 2*n/3};
    PatchEncoding<OutsideOperator<int>> encoding(op);
    Benchmark_Decoding(encoding, data, DataType::d_int, state);
}
BENCHMARK(BM_Patch_ConsecutiveNumbers_OutsideOperator_Int_Decode)->Arg(1<<15)->Arg(1<<20);

} /* namespace ddj */
