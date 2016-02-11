/*
 * gfc_encoding_benchmark.cpp
 *
 *  Created on: Nov 17, 2015
 *      Author: Karol Dzitkowski
 */

#include "benchmarks/encoding_benchmark_base.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "compression/gfc/gfc_encoding.hpp"

namespace ddj
{

class GfcEncodingBenchmark : public EncodingBenchmarkBase {};

BENCHMARK_DEFINE_F(GfcEncodingBenchmark, BM_Gfc_Random_Double_Encode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<double, char>(CudaArrayGenerator().GenerateRandomDoubleDeviceArray(n));
    GfcEncoding encoding;
    Benchmark_Encoding(encoding, data, DataType::d_double, state);
}
BENCHMARK_REGISTER_F(GfcEncodingBenchmark, BM_Gfc_Random_Double_Encode)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24);

BENCHMARK_DEFINE_F(GfcEncodingBenchmark, BM_Gfc_Random_Float_Decode)(benchmark::State& state)
{
    int n = state.range_x();
    auto data = MoveSharedCudaPtr<double, char>(CudaArrayGenerator().GenerateRandomDoubleDeviceArray(n));
    GfcEncoding encoding;
    Benchmark_Encoding(encoding, data, DataType::d_double, state);
}
BENCHMARK_REGISTER_F(GfcEncodingBenchmark, BM_Gfc_Random_Float_Decode)->Arg(1<<22)->Arg(1<<23)->Arg(1<<24);


} /* namespace ddj */



