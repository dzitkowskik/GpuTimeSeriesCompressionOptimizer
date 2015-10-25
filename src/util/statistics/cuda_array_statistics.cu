/*
 *  cuda_array_statistics.cu
 *
 *  Created on: 21-10-2015
 *      Author: Karol Dzitkowski
 */

#include "util/statistics/cuda_array_statistics.hpp"
#include "core/cuda_macros.cuh"
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

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

#define CUDA_ARRAY_STATISTICS_SPEC(X) \
	template std::tuple<X,X> CudaArrayStatistics::MinMax<X>(SharedCudaPtr<X>); \
	template char CudaArrayStatistics::MinBitCnt<X>(SharedCudaPtr<X>);
FOR_EACH(CUDA_ARRAY_STATISTICS_SPEC, float, int, long long, unsigned int)

} /* namespace ddj */
