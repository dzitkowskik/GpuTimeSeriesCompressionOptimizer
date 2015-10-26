#include "benchmarks/encoding_benchmark_base.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "compression/dict/dict_encoding.hpp"

namespace ddj
{

class DictEncodingBenchmark : public EncodingBenchmarkBase {};

BENCHMARK_DEFINE_F(DictEncodingBenchmark, BM_Dict_Random_Int_Encode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int, char>(CudaArrayGenerator().GenerateRandomIntDeviceArray(n));
    DictEncoding encoding;
    Benchmark_Encoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(DictEncodingBenchmark, BM_Dict_Random_Int_Encode)->Arg(1<<15)->Arg(1<<20);

BENCHMARK_DEFINE_F(DictEncodingBenchmark, BM_Dict_Random_Int_Decode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int, char>(CudaArrayGenerator().GenerateRandomIntDeviceArray(n));
    DictEncoding encoding;
    Benchmark_Decoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(DictEncodingBenchmark, BM_Dict_Random_Int_Decode)->Arg(1<<15)->Arg(1<<20);

BENCHMARK_DEFINE_F(DictEncodingBenchmark, BM_Dict_Random_Float_Encode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<float, char>(CudaArrayGenerator().GenerateRandomFloatDeviceArray(n));
    DictEncoding encoding;
    Benchmark_Encoding(encoding, data, DataType::d_float, state);
}
BENCHMARK_REGISTER_F(DictEncodingBenchmark, BM_Dict_Random_Float_Encode)->Arg(1<<15)->Arg(1<<20);

BENCHMARK_DEFINE_F(DictEncodingBenchmark, BM_Dict_Random_Float_Decode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<float, char>(CudaArrayGenerator().GenerateRandomFloatDeviceArray(n));
    DictEncoding encoding;
    Benchmark_Decoding(encoding, data, DataType::d_float, state);
}
BENCHMARK_REGISTER_F(DictEncodingBenchmark, BM_Dict_Random_Float_Decode)->Arg(1<<15)->Arg(1<<20);

} /* namespace ddj */
