#include "benchmarks/encoding_benchmark_base.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "compression/afl/afl_encoding.hpp"
#include "compression/afl/afl_gpu.cuh"

namespace ddj
{

class AflEncodingBenchmark : public EncodingBenchmarkBase {};

BENCHMARK_DEFINE_F(AflEncodingBenchmark, BM_Afl_Random_Int_Encode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int, char>(CudaArrayGenerator().GenerateRandomIntDeviceArray(n));
    AflEncoding encoding;
    Benchmark_Encoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(AflEncodingBenchmark, BM_Afl_Random_Int_Encode)
	->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(AflEncodingBenchmark, BM_Afl_Random_Int_Decode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<int, char>(CudaArrayGenerator().GenerateRandomIntDeviceArray(n));
    AflEncoding encoding;
    Benchmark_Decoding(encoding, data, DataType::d_int, state);
}
BENCHMARK_REGISTER_F(AflEncodingBenchmark, BM_Afl_Random_Int_Decode)
	->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(AflEncodingBenchmark, BM_Pure_Afl_Aligned)(benchmark::State& state)
{
	auto data = CudaArrayGenerator().GetFakeDataWithPatternA<int>(0, 10, 1, 1, 1000, state.range_x());

	// Get minimal bit count needed to encode data
	char minBit = 10;
	int elemBitSize = 8*sizeof(int);
	int comprElemCnt = (minBit * data->size() + elemBitSize - 1) / elemBitSize;
	int comprDataSize = comprElemCnt * sizeof(int);
	auto result = CudaPtr<char>::make_shared(comprDataSize);

	while (state.KeepRunning())
	{
		run_afl_compress_gpu<int, 32>(
				minBit,
				data->get(),
				(int*)result->get(),
				data->size(),
				comprDataSize/sizeof(int));
	}
	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(AflEncodingBenchmark, BM_Pure_Afl_Aligned)
	->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

BENCHMARK_DEFINE_F(AflEncodingBenchmark, BM_Pure_Afl_NonAligned)(benchmark::State& state)
{
	auto data = CudaArrayGenerator().GetFakeDataWithPatternA<int>(0, 10, 1, 1, 1000, state.range_x());

	// Get minimal bit count needed to encode data
	char minBit = 10;
	int elemBitSize = 8*sizeof(int);
	int comprElemCnt = (minBit * data->size() + elemBitSize - 1) / elemBitSize;
	int comprDataSize = comprElemCnt * sizeof(int);
	auto result = CudaPtr<char>::make_shared(comprDataSize);

	while (state.KeepRunning())
	{
		run_afl_compress_gpu<int, 1>(
				minBit,
				data->get(),
				(int*)result->get(),
				data->size(),
				comprDataSize/sizeof(int));
	}
	SetStatistics(state, DataType::d_int);
}
BENCHMARK_REGISTER_F(AflEncodingBenchmark, BM_Pure_Afl_NonAligned)
	->Arg(1<<21)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24)->Arg(1<<25);

} /* namespace ddj */
