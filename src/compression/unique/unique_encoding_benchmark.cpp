#include "util/generator/cuda_array_generator.hpp"
#include "compression/unique/unique_encoding.hpp"
#include "benchmarks/encoding_benchmark_base.hpp"

namespace ddj {

class UniqueEncodingBenchmark : public EncodingBenchmarkBase {};

BENCHMARK_DEFINE_F(UniqueEncodingBenchmark, BM_Unique_Random_Int_Encode)(benchmark::State& state)
{
	UniqueEncoding encoding;
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int,char>(
    		CudaArrayGenerator().GenerateRandomIntDeviceArray(n,0,3));
    Benchmark_Encoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(UniqueEncodingBenchmark, BM_Unique_Random_Int_Encode)->Arg(1<<21)->Arg(1<<22)->Arg(1<<23);

BENCHMARK_DEFINE_F(UniqueEncodingBenchmark, BM_Unique_Random_Int_Decode)(benchmark::State& state)
{
	UniqueEncoding encoding;
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int,char>(
    		CudaArrayGenerator().GenerateRandomIntDeviceArray(n,0,3));
    Benchmark_Decoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(UniqueEncodingBenchmark, BM_Unique_Random_Int_Decode)->Arg(1<<21)->Arg(1<<22)->Arg(1<<23);

} /* namespace ddj */
