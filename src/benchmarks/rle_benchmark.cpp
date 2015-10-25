#include "util/generator/cuda_array_generator.hpp"
#include "compression/rle/rle_encoding.hpp"
#include "templates.hpp"

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

namespace ddj {

static void BM_Rle_Random_Float_Encode(benchmark::State& state)
{
    RleEncoding encoding;
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<float,char>(
    		CudaArrayGenerator().GenerateRandomFloatDeviceArray(n));
    Benchmark_Encoding(encoding, data, DataType::d_float, state);
}
BENCHMARK(BM_Rle_Random_Float_Encode)->Arg(1<<15)->Arg(1<<20);

static void BM_Rle_Random_Float_Decode(benchmark::State& state)
{
    RleEncoding encoding;
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<float,char>(
    		CudaArrayGenerator().GenerateRandomFloatDeviceArray(n));
    Benchmark_Decoding(encoding, data, DataType::d_float, state);
}
BENCHMARK(BM_Rle_Random_Float_Decode)->Arg(1<<15)->Arg(1<<20);

} /* namespace ddj */
