/*
 * scale_encoding.cpp
 *
 *  Created on: 24-04-2015
 *      Author: Karol Dzitkowski
 */

#include "scale_encoding.hpp"
#include "helpers/helper_macros.h"
#include "core/cuda_macros.cuh"

#include <cuda_runtime_api.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>


#define SCALE_ENCODING_GPU_BLOCK_SIZE 64
#define SCALE_DECODING_GPU_BLOCK_SIZE 64

namespace ddj {

template<typename T>
__global__ void scaleEncodeKernel(T* data, int size, T* result, T min)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= size) return;
	result[idx] = data[idx] - min;
}

template<typename T>
SharedCudaPtr<char> ScaleEncoding::Encode(SharedCudaPtr<T> data)
{
	int block_size = SCALE_ENCODING_GPU_BLOCK_SIZE;
	int block_cnt = (data->size() + block_size - 1) / block_size;

	thrust::device_ptr<T> data_ptr(data->get());
	auto result = CudaPtr<char>::make_shared((data->size()+1)*sizeof(T));
	T min = thrust::min_element(data_ptr, data_ptr+data->size())[0];
	CUDA_CALL( cudaMemcpy(result->get(), &min, sizeof(T), cudaMemcpyHostToDevice) );

	scaleEncodeKernel<T><<<block_size, block_cnt>>>(
			data->get(),
			data->size(),
			(T*)(result->get()+sizeof(T)),
			min);
	cudaDeviceSynchronize();

	return result;
}

template<typename T>
__global__ void scaleDecodeKernel(T* data, int size, T* result, T min)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= size) return;
	result[idx] = data[idx] + min;
}

template<typename T>
SharedCudaPtr<T> ScaleEncoding::Decode(SharedCudaPtr<char> data)
{
	int size = data->size()/sizeof(T)-1;
	int block_size = SCALE_ENCODING_GPU_BLOCK_SIZE;
	int block_cnt = (data->size() + block_size - 1) / block_size;

	thrust::device_ptr<T> data_ptr((T*)data->get());
	auto min = data_ptr[0];

	auto result = CudaPtr<T>::make_shared(size);
	scaleDecodeKernel<T><<<block_size, block_cnt>>>(
			(T*)(data->get()+sizeof(T)), size, result->get(), min);
	cudaDeviceSynchronize();

	return result;
}

#define SCALE_SPEC(X) \
	template SharedCudaPtr<char> ScaleEncoding::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> ScaleEncoding::Decode<X>(SharedCudaPtr<char> data);
FOR_EACH(SCALE_SPEC, double, float, int, long, long long, unsigned int, unsigned long, unsigned long long)

} /* namespace ddj */
