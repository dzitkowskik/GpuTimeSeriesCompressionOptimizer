/*
 *  delta_encoding.cu
 *
 *  Created on: 18-04-2015
 *      Author: Karol Dzitkowski
 */

#include "delta_encoding.hpp"
#include "compression/encoding_type.hpp"
#include "helpers/helper_macros.h"
#include "helpers/helper_cuda.cuh"
#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define DELTA_ENCODING_GPU_BLOCK_SIZE 64
#define DELTA_DECODING_GPU_BLOCK_SIZE 64

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

// CREATE METADATA
// TODO: Move to deltaEncodeKernel kernel and check performance
template<typename T>
SharedCudaPtr<char> EncodeMetadata(SharedCudaPtr<T> data)
{
	EncodingType type = EncodingType::delta;
	auto result = CudaPtr<char>::make_shared(sizeof(EncodingType) + sizeof(T));
	CUDA_CALL( cudaMemcpy(result->get(), &type, sizeof(EncodingType), cudaMemcpyHostToDevice) );
	CUDA_CALL( cudaMemcpy(result->get()+sizeof(EncodingType), data->get(), sizeof(T), cudaMemcpyDeviceToDevice) );

	return result;
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

	auto result_metadata = EncodeMetadata<T>(data);
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
	T* first_value = (T*)(metadata->get()+sizeof(EncodingType));
	this->_policy.setSize(size);
	cudaLaunch(this->_policy, addValueKernel<T>,
		result->get(),
		size,
		first_value
	);

	return result;
}

#define SCALE_SPEC(X) \
	template SharedCudaPtrVector<char> DeltaEncoding::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> DeltaEncoding::Decode<X>(SharedCudaPtrVector<char> data);
FOR_EACH(SCALE_SPEC, double, float, int, long, long long, unsigned int, unsigned long, unsigned long long)


} /* namespace ddj */
