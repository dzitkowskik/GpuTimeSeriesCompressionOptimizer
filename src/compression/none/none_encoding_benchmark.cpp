#include "util/generator/cuda_array_generator.hpp"
#include "compression/none/none_encoding.hpp"
#include "benchmarks/encoding_benchmark_base.hpp"

namespace ddj {

class NoneEncodingBenchmark : public EncodingBenchmarkBase {};

BENCHMARK_DEFINE_F(NoneEncodingBenchmark, BM_None_Random_Int_Encode)(benchmark::State& state)
{
	NoneEncoding encoding;
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int,char>(
    		CudaArrayGenerator().GenerateRandomIntDeviceArray(n));
    Benchmark_Encoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(NoneEncodingBenchmark, BM_None_Random_Int_Encode)
	->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(NoneEncodingBenchmark, BM_None_Random_Int_Decode)(benchmark::State& state)
{
	NoneEncoding encoding;
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int,char>(
    		CudaArrayGenerator().GenerateRandomIntDeviceArray(n));
    Benchmark_Decoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(NoneEncodingBenchmark, BM_None_Random_Int_Decode)
	->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(NoneEncodingBenchmark, BM_None_Random_Float_Encode)(benchmark::State& state)
{
	NoneEncoding encoding;
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<float,char>(
    		CudaArrayGenerator().GenerateRandomFloatDeviceArray(n));
    Benchmark_Encoding(encoding, data, DataType::d_float, state);
}
BENCHMARK_REGISTER_F(NoneEncodingBenchmark, BM_None_Random_Float_Encode)
	->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(NoneEncodingBenchmark, BM_None_Random_Float_Decode)(benchmark::State& state)
{
	NoneEncoding encoding;
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<float,char>(
    		CudaArrayGenerator().GenerateRandomFloatDeviceArray(n));
    Benchmark_Decoding(encoding, data, DataType::d_float, state);
}
BENCHMARK_REGISTER_F(NoneEncodingBenchmark, BM_None_Random_Float_Decode)
	->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

} /* namespace ddj */
