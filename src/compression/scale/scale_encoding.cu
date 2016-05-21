/*
 * scale_encoding.cpp
 *
 *  Created on: 24-04-2015
 *      Author: Karol Dzitkowski
 */

#include "scale_encoding.hpp"
#include "core/macros.h"
#include "core/cuda_macros.cuh"
#include "core/cuda_launcher.cuh"
#include "util/statistics/cuda_array_statistics.hpp"
#include <cuda_runtime_api.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <limits>
#include <cmath>

namespace ddj {

template<typename T>
__global__ void scaleEncodeKernel(T* data, int size, T* result_data, T min, T* result_metadata)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= size) return;
	result_data[idx] = data[idx] - min;
	result_metadata[0] = min;
}

template<typename T>
SharedCudaPtrVector<char> ScaleEncoding::Encode(SharedCudaPtr<T> data)
{
	CUDA_ASSERT_RETURN( cudaGetLastError() );
    LOG4CPLUS_INFO_FMT(_logger, "SCALE encoding START: data size = %lu", data->size());

	if(data->size() <= 0)
		return SharedCudaPtrVector<char>{ CudaPtr<char>::make_shared(), CudaPtr<char>::make_shared() };

	// ALLOCATE RESULTS
	auto result_data = CudaPtr<char>::make_shared(data->size()*sizeof(T));
	auto result_metadata = CudaPtr<char>::make_shared(sizeof(T));

	// GET MIN VALUE OF DATA
	auto minMax = CudaArrayStatistics().MinMax(data);

	// TAKE ABSOLUTE VALUE
	T min = std::get<0>(minMax);
	T max = std::get<1>(minMax);
	if(min < 0)	// overflow detection
		if((std::numeric_limits<T>::max() - max) < std::abs<T>(min)) min = 0;

	LOG4CPLUS_TRACE(_logger, "SCALE min = " << min);

	// SCALE DATA BY MIN VALUE
	this->_policy.setSize(data->size());
	cudaLaunch(this->_policy, scaleEncodeKernel<T>,
			data->get(),
			data->size(),
			(T*)result_data->get(),
			min,
			(T*)result_metadata->get());
	cudaDeviceSynchronize();

	CUDA_ASSERT_RETURN( cudaGetLastError() );
	LOG4CPLUS_INFO(_logger, "SCALE enoding END");

	return SharedCudaPtrVector<char> {result_metadata, result_data};
}

template<typename T>
__global__ void scaleDecodeKernel(T* data, int size, T* result, T* min)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= size) return;
	result[idx] = data[idx] + *min;
}

template<typename T>
SharedCudaPtr<T> ScaleEncoding::Decode(SharedCudaPtrVector<char> input)
{
	LOG4CPLUS_INFO_FMT(
		_logger,
		"SCALE decoding START: input[0] size = %lu, input[1] size = %lu",
		input[0]->size(), input[1]->size()
	);

	if(input[1]->size() <= 0)
		return CudaPtr<T>::make_shared();

	auto metadata = input[0];
	auto data = input[1];

	int size = data->size()/sizeof(T);
	auto result = CudaPtr<T>::make_shared(size);

	this->_policy.setSize(size);
	cudaLaunch(this->_policy, scaleDecodeKernel<T>,
			(T*)data->get(),
			size,
			result->get(),
			(T*)metadata->get());
	cudaDeviceSynchronize();

	CUDA_ASSERT_RETURN( cudaGetLastError() );
	LOG4CPLUS_INFO(_logger, "SCALE decoding END");

	return result;
}

#define SCALE_SPEC(X) \
	template SharedCudaPtrVector<char> ScaleEncoding::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> ScaleEncoding::Decode<X>(SharedCudaPtrVector<char> data);
FOR_EACH(SCALE_SPEC, char, short, double, float, int, long, long long, unsigned int)

} /* namespace ddj */
