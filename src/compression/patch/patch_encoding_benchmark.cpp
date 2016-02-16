#include "util/generator/cuda_array_generator.hpp"
#include "compression/patch/outside_patch_encoding.hpp"
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
    OutsidePatchEncoding encoding(0, n, 0.2);
    Benchmark_Encoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(PatchEncodingBenchmark, BM_Patch_ConsecutiveNumbers_OutsideOperator_Int_Encode)
	->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);;

BENCHMARK_DEFINE_F(PatchEncodingBenchmark, BM_Patch_ConsecutiveNumbers_OutsideOperator_Int_Decode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int, char>(
    		CudaArrayGenerator().GenerateConsecutiveIntDeviceArray(n));
    OutsidePatchEncoding encoding(0, n, 0.2);
    Benchmark_Decoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(PatchEncodingBenchmark, BM_Patch_ConsecutiveNumbers_OutsideOperator_Int_Decode)
	->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(PatchEncodingBenchmark, BM_Patch_Random_OutsideOperator_Int_Encode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int, char>(
    		CudaArrayGenerator().GenerateRandomIntDeviceArray(n, 10, 1000));
    OutsidePatchEncoding encoding(10, 1000);
    Benchmark_Decoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(PatchEncodingBenchmark, BM_Patch_Random_OutsideOperator_Int_Encode)
->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(PatchEncodingBenchmark, BM_Patch_Random_OutsideOperator_Int_Decode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int, char>(
    		CudaArrayGenerator().GenerateRandomIntDeviceArray(n, 10, 1000));
    OutsidePatchEncoding encoding(10, 1000);
    Benchmark_Decoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(PatchEncodingBenchmark, BM_Patch_Random_OutsideOperator_Int_Decode)
->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(PatchEncodingBenchmark, BM_Patch_ConsecutiveNumbers_OutsideOperator_Float_Encode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<float, char>(
    		CudaArrayGenerator().GenerateRandomFloatDeviceArray(n));
    OutsidePatchEncoding encoding(0, n, 0.25);
    Benchmark_Encoding(encoding, data, DataType::d_float, state);
}
BENCHMARK_REGISTER_F(PatchEncodingBenchmark, BM_Patch_ConsecutiveNumbers_OutsideOperator_Float_Encode)
->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(PatchEncodingBenchmark, BM_Patch_ConsecutiveNumbers_OutsideOperator_Float_Decode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<float, char>(
    		CudaArrayGenerator().GenerateRandomFloatDeviceArray(n));
    OutsidePatchEncoding encoding(0, n, 0.25);
    Benchmark_Decoding(encoding, data, DataType::d_float, state);
}
BENCHMARK_REGISTER_F(PatchEncodingBenchmark, BM_Patch_ConsecutiveNumbers_OutsideOperator_Float_Decode)
->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

} /* namespace ddj */
