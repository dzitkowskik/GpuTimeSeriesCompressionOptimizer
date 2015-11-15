/*
 *  cuda_array_statistics.cu
 *
 *  Created on: 21-10-2015
 *      Author: Karol Dzitkowski
 */

#include "util/statistics/cuda_array_statistics.hpp"
#include "core/cuda_macros.cuh"
#include "core/macros.h"
#include "core/cuda_launcher.cuh"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <math_functions.h>

namespace ddj {

template<typename T> std::tuple<T,T> CudaArrayStatistics::MinMax(SharedCudaPtr<T> data)
{
	thrust::device_ptr<T> dp(data->get());
	auto tuple = thrust::minmax_element(dp, dp+data->size());
	T min = *(tuple.first);
	T max = *(tuple.second);
	return std::make_tuple(min, max);
}

template<typename T>
char CudaArrayStatistics::MinBitCnt(SharedCudaPtr<T> data)
{
	auto minMax = MinMax(data);
	int result = 32;
	if (std::get<0>(minMax) >= 0)
		result = ALT_BITLEN(std::get<1>(minMax));
	return result;
}


__host__ __device__ int _getFloatPrecision(float number)
{
	long int e = 1;
	while(round(number*e)/e != number) e*=10;
	return logf(e) / logf(10);
}

template<typename T>
__global__ void _precisionKernelFloat(T* input, size_t size, int* output)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	output[idx] = _getFloatPrecision(input[idx]);
}

template<typename T>
int CudaArrayStatistics::Precision(SharedCudaPtr<T> data)
{
	auto precisions = CudaPtr<int>::make_shared(data->size());

	this->_policy.setSize(data->size());
	cudaLaunch(this->_policy, _precisionKernelFloat<T>,
			data->get(),
			data->size(),
			precisions->get());

	auto minMax = this->MinMax(precisions);
	return std::get<1>(minMax);
}

#define CUDA_ARRAY_STATISTICS_SPEC(X) \
	template std::tuple<X,X> CudaArrayStatistics::MinMax<X>(SharedCudaPtr<X>); \
	template char CudaArrayStatistics::MinBitCnt<X>(SharedCudaPtr<X>); \
	template int CudaArrayStatistics::Precision<X>(SharedCudaPtr<X>);
FOR_EACH(CUDA_ARRAY_STATISTICS_SPEC, float, int, long, long long, unsigned int)

} /* namespace ddj */
