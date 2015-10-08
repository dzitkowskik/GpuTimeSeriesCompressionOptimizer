/*
 *  delta_encoding.cu
 *
 *  Created on: 18-04-2015
 *      Author: Karol Dzitkowski
 */

#include "delta_encoding.hpp"
#include "helpers/helper_macros.h"
#include "helpers/helper_cuda.cuh"

#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace ddj {

template<typename T>
__global__ void deltaEncodeKernel(T* data, int size, T* result)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx+1 >= size) return;
	register T v1 = data[idx];
	register T v2 = data[idx+1];
	result[idx] = v2 - v1;
}

template<typename T>
SharedCudaPtrVector<char> DeltaEncoding::Encode(SharedCudaPtr<T> data)
{
	// MAKE DELTA ENCODING
	auto result_data = CudaPtr<char>::make_shared((data->size() - 1) * sizeof(T));
	this->_policy.setSize(data->size());
	cudaLaunch(this->_policy, deltaEncodeKernel<T>,
		data->get(),
		data->size(),
		(T*)(result_data->get())
	);

	// SAVE FIRST VALUE TO METADATA
	auto result_metadata = CudaPtr<char>::make_shared(sizeof(T));
	CUDA_CALL( cudaMemcpy(result_metadata->get(), data->get(), sizeof(T), cudaMemcpyDeviceToDevice) );

	cudaDeviceSynchronize();

	return SharedCudaPtrVector<char> {result_metadata, result_data};
}

template<typename T>
__global__ void addValueKernel(T* data, const int size, T* value)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= size) return;
	data[idx] += *value;
}

template<typename T>
SharedCudaPtr<T> DeltaEncoding::Decode(SharedCudaPtrVector<char> input)
{
	auto metadata = input[0];
	auto data = input[1];

	int size = data->size()/sizeof(T) + 1;
	auto result = CudaPtr<T>::make_shared(size);

	// Calculate deltas
	thrust::device_ptr<T> data_ptr((T*)data->get());
	thrust::device_ptr<T> result_ptr(result->get());
	thrust::inclusive_scan(data_ptr, data_ptr+(size-1), result_ptr+1);
	result_ptr[0] = 0;

	// Add first value all elements
	this->_policy.setSize(size);
	cudaLaunch(this->_policy, addValueKernel<T>,
		result->get(),
		size,
		(T*)metadata->get()
	);

	return result;
}

#define DELTA_ENCODING_SPEC(X) \
	template SharedCudaPtrVector<char> DeltaEncoding::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> DeltaEncoding::Decode<X>(SharedCudaPtrVector<char> data);
FOR_EACH(DELTA_ENCODING_SPEC, float, int, long long, unsigned int)


} /* namespace ddj */
