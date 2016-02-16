#include "benchmarks/encoding_benchmark_base.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "compression/float/float_encoding.hpp"

namespace ddj
{

class FloatEncodingBenchmark : public EncodingBenchmarkBase {};

BENCHMARK_DEFINE_F(FloatEncodingBenchmark, BM_Float_Random_Float_Encode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<float, char>(GetFloatRandomDataWithMaxPrecision(n, 3));
    FloatEncoding encoding;
    Benchmark_Encoding(encoding, data, DataType::d_float, state);
}
BENCHMARK_REGISTER_F(FloatEncodingBenchmark, BM_Float_Random_Float_Encode)
	->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(FloatEncodingBenchmark, BM_Float_Random_Float_Decode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<float, char>(GetFloatRandomDataWithMaxPrecision(n, 3));
    FloatEncoding encoding;
    Benchmark_Decoding(encoding, data, DataType::d_float, state);
}
BENCHMARK_REGISTER_F(FloatEncodingBenchmark, BM_Float_Random_Float_Decode)
	->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

} /* namespace ddj */
