#include "util/generator/cuda_array_generator.hpp"
#include "compression/patch/patch_encoding.hpp"
#include "util/stencil/stencil.hpp"
#include "benchmarks/encoding_benchmark_base.hpp"

namespace ddj
{

class PatchEncodingBenchmark : public EncodingBenchmarkBase {};

BENCHMARK_DEFINE_F(PatchEncodingBenchmark, BM_Patch_ConsecutiveNumbers_OutsideOperator_Int_Encode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int, char>(
    		CudaArrayGenerator().GenerateConsecutiveIntDeviceArray(n));
    OutsideOperator<int> op{n/3, 2*n/3};
    PatchEncoding<OutsideOperator<int>> encoding(op);
    Benchmark_Encoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(PatchEncodingBenchmark, BM_Patch_ConsecutiveNumbers_OutsideOperator_Int_Encode)->Arg(1<<15)->Arg(1<<20);

BENCHMARK_DEFINE_F(PatchEncodingBenchmark, BM_Patch_ConsecutiveNumbers_OutsideOperator_Int_Decode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int, char>(
    		CudaArrayGenerator().GenerateConsecutiveIntDeviceArray(n));
    OutsideOperator<int> op{n/3, 2*n/3};
    PatchEncoding<OutsideOperator<int>> encoding(op);
    Benchmark_Decoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(PatchEncodingBenchmark, BM_Patch_ConsecutiveNumbers_OutsideOperator_Int_Decode)->Arg(1<<15)->Arg(1<<20);

BENCHMARK_DEFINE_F(PatchEncodingBenchmark, BM_Patch_ConsecutiveNumbers_OutsideOperator_Float_Encode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<float, char>(
    		CudaArrayGenerator().GenerateRandomFloatDeviceArray(n));
    OutsideOperator<float> op{n/3.0f, 2.0f*n/3.0f};
    PatchEncoding<OutsideOperator<float>> encoding(op);
    Benchmark_Encoding(encoding, data, DataType::d_float, state);
}
BENCHMARK_REGISTER_F(PatchEncodingBenchmark, BM_Patch_ConsecutiveNumbers_OutsideOperator_Float_Encode)->Arg(1<<15)->Arg(1<<20);

BENCHMARK_DEFINE_F(PatchEncodingBenchmark, BM_Patch_ConsecutiveNumbers_OutsideOperator_Float_Decode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<float, char>(
    		CudaArrayGenerator().GenerateRandomFloatDeviceArray(n));
    OutsideOperator<float> op{n/3.0f, 2.0f*n/3.0f};
    PatchEncoding<OutsideOperator<float>> encoding(op);
    Benchmark_Decoding(encoding, data, DataType::d_float, state);
}
BENCHMARK_REGISTER_F(PatchEncodingBenchmark, BM_Patch_ConsecutiveNumbers_OutsideOperator_Float_Decode)->Arg(1<<15)->Arg(1<<20);

} /* namespace ddj */
