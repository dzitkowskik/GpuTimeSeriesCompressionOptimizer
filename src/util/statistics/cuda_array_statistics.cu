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
#include "util/other/cuda_array_reduce.cuh"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <math_functions.h>

namespace ddj {

template<typename T>
std::tuple<T,T> CudaArrayStatistics::MinMax(SharedCudaPtr<T> data)
{
	thrust::device_ptr<T> dp(data->get());
	auto tuple = thrust::minmax_element(dp, dp+data->size());
	T min = *(tuple.first);
	T max = *(tuple.second);
	return std::make_tuple(min, max);
}

template<typename T>
char getMinBit(T min, T max)
{
	int result = sizeof(T)*8;
	if (min >= 0)
		result = ALT_BITLEN(max);
	return result;

}

template<typename T>
char CudaArrayStatistics::MinBitCnt(SharedCudaPtr<T> data)
{
	auto minMax = MinMax(data);
	return getMinBit(std::get<0>(minMax), std::get<1>(minMax));
}

__host__ __device__
int _getFloatPrecision(float number)
{
	long int e = 1;
	long int maxPrecision = pow(10, MAX_PRECISION);
	while(round(number*e)/e != number && e < maxPrecision) e*=10;
	return logf(e) / logf(10);
}

template<typename T> __global__
void _precisionKernelFloat(T* input, size_t size, int* output)
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

template<typename T> __global__
void _sortedKernel(T* data, size_t size, bool* result)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= size-1) return;
	if(data[idx] > data[idx+1]) *result = true;
}


template<typename T>
bool CudaArrayStatistics::Sorted(SharedCudaPtr<T> data)
{
	auto result = CudaPtr<bool>::make_shared(1);
	CUDA_CALL( cudaMemset(result->get(), 0, sizeof(bool)) );

	this->_policy.setSize(data->size());
	cudaLaunch(this->_policy, _sortedKernel<T>,
			data->get(),
			data->size(),
			result->get());

	bool h_result;
	CUDA_CALL( cudaMemcpy(&h_result, result->get(), sizeof(bool), CPY_DTH) );
	return !h_result;
}



template<typename T, int N=3> __global__
void _rlMetricKernel(T* data, size_t size, int* result)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= size-N) return;

	int value = data[idx];
	int num = 1, out = 1, last = 1;

	#pragma unroll
	for(int i = 1; i < N; i++)
	{
		out = value == data[idx+i];
		num += out & last;
		last = out;
	}

	result[idx] = num;
}


template<typename T, int N>
float CudaArrayStatistics::RlMetric(SharedCudaPtr<T> data)
{
	if(N >= data->size()) return 1.0;
	size_t size = data->size() - N;
	auto result = CudaPtr<int>::make_shared(size);
	this->_policy.setSize(size);
	cudaLaunch(this->_policy, _rlMetricKernel<T,N>,
			data->get(),
			data->size(),
			result->get());

	auto sum = reduce_thrust(result, thrust::plus<int>());
	return sum / size;
}

template<typename T>
T CudaArrayStatistics::Mean(SharedCudaPtr<T> data)
{
	auto sum = reduce_thrust(data, thrust::plus<T>());
	return sum / data->size();
}

template<typename T>
DataStatistics CudaArrayStatistics::getStatistics(SharedCudaPtr<T> data)
{
	// printf("Get statistics for data with size = %lu\n", data->size());
	if(data->size() <= 0) return DataStatistics();
	DataStatistics stats;
	auto minMax = MinMax(data);
	stats.min = std::get<0>(minMax);
	stats.max = std::get<1>(minMax);
	stats.minBitCnt = getMinBit(stats.min, stats.max);
	stats.precision = Precision(data);
	stats.sorted = Sorted(data);
	stats.rlMetric = RlMetric(data);
	stats.mean = Mean(data);
	return stats;
}

// TODO: Make this a common template in header file and use everywhere
DataStatistics CudaArrayStatistics::GenerateStatistics(SharedCudaPtr<char> data, DataType type)
{
	// printf("Generating statistics for data with size = %lu\n", data->size());
	switch(type)
	{
		case DataType::d_int:
			return getStatistics(boost::reinterpret_pointer_cast<CudaPtr<int>>(data));
		case DataType::d_time:
			return getStatistics(boost::reinterpret_pointer_cast<CudaPtr<time_t>>(data));
		case DataType::d_float:
			return getStatistics(boost::reinterpret_pointer_cast<CudaPtr<float>>(data));
		case DataType::d_double:
			return getStatistics(boost::reinterpret_pointer_cast<CudaPtr<double>>(data));
		default:
			throw NotImplementedException("No CudaArrayStatistics::GenerateStatistics implementation for that type");
	}
}

#define CUDA_ARRAY_STATISTICS_SPEC(X) \
	template std::tuple<X,X> CudaArrayStatistics::MinMax<X>(SharedCudaPtr<X>); \
	template char CudaArrayStatistics::MinBitCnt<X>(SharedCudaPtr<X>); \
	template int CudaArrayStatistics::Precision<X>(SharedCudaPtr<X>); \
    template bool CudaArrayStatistics::Sorted<X>(SharedCudaPtr<X>); \
    template float CudaArrayStatistics::RlMetric<X,2>(SharedCudaPtr<X>); \
    template float CudaArrayStatistics::RlMetric<X,3>(SharedCudaPtr<X>); \
    template float CudaArrayStatistics::RlMetric<X,4>(SharedCudaPtr<X>); \
    template float CudaArrayStatistics::RlMetric<X,5>(SharedCudaPtr<X>); \
    template float CudaArrayStatistics::RlMetric<X,6>(SharedCudaPtr<X>); \
    template X CudaArrayStatistics::Mean<X>(SharedCudaPtr<X> data);
FOR_EACH(CUDA_ARRAY_STATISTICS_SPEC, char, short, double, float, int, long, long long, unsigned int)

} /* namespace ddj */
