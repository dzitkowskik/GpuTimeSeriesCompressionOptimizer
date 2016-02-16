#include "benchmarks/encoding_benchmark_base.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "compression/const/const_encoding.hpp"

namespace ddj
{

class ConstEncodingBenchmark : public EncodingBenchmarkBase {};

BENCHMARK_DEFINE_F(ConstEncodingBenchmark, BM_Const_Random_Int_Encode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int, char>(CudaArrayGenerator().GenerateRandomIntDeviceArray(n));
    ConstEncoding encoding;
    Benchmark_Encoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(ConstEncodingBenchmark, BM_Const_Random_Int_Encode)
	->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(ConstEncodingBenchmark, BM_Const_Random_Int_Decode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int, char>(CudaArrayGenerator().GenerateRandomIntDeviceArray(n));
    ConstEncoding encoding;
    Benchmark_Decoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(ConstEncodingBenchmark, BM_Const_Random_Int_Decode)
	->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

} /* namespace ddj */
