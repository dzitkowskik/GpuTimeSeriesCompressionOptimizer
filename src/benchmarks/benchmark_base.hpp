/*
 *  benchmark_base.hpp
 *
 *  Created on: 11/11/2015
 *      Author: Karol Dzitkowski
 */

#ifndef BENCHMARK_BASE_HPP_
#define BENCHMARK_BASE_HPP_

#include "core/cuda_ptr.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "time_series.h"

#include <benchmark/benchmark.h>

namespace ddj
{

class BenchmarkBase : public benchmark::Fixture
{
protected:
	SharedCudaPtr<int> GetIntRandomData(int n, int from = 100, int to = 1000);
	SharedCudaPtr<int> GetIntConsecutiveData(int n);
	SharedCudaPtr<float> GetFloatRandomData(int n);
	SharedCudaPtr<float> GetFloatRandomDataWithMaxPrecision(int n, int maxPrecision);
	SharedCudaPtr<time_t> GetTsIntDataFromFile(int n);
	SharedCudaPtr<float> GetTsFloatDataFromFile(int n);
	SharedCudaPtr<int> GetFakeIntDataForHistogram(int n);

protected:
	CudaArrayGenerator _generator;
};

} /* namespace ddj */
#endif /* BENCHMARK_BASE_HPP_ */
