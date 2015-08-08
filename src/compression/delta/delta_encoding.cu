/*
 * delta_encoding.cu
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

template<typename T>
SharedCudaPtr<char> DeltaEncoding::Encode(SharedCudaPtr<T> data)
{
	auto result = CudaPtr<char>::make_shared(data->size()*sizeof(T));

	this->policy.setSize(data->size());
	cudaLaunch(this->policy, deltaEncodeKernel<T>,
		data->get(), data->size(), (T*)(result->get()+sizeof(T)));

	CUDA_CALL( cudaMemcpy(result->get(), data->get(), sizeof(T), cudaMemcpyDeviceToDevice) );
	cudaDeviceSynchronize();

	return result;
}

template<typename T>
SharedCudaPtr<T> DeltaEncoding::Decode(SharedCudaPtr<char> data)
{
	int size = data->size()/sizeof(T);
	auto result = CudaPtr<T>::make_shared(size);

	thrust::device_ptr<T> data_ptr((T*)data->get());
	thrust::device_ptr<T> result_ptr(result->get());
	thrust::inclusive_scan(data_ptr, data_ptr+size, result_ptr);

	return result;
}

#define SCALE_SPEC(X) \
	template SharedCudaPtr<char> DeltaEncoding::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> DeltaEncoding::Decode<X>(SharedCudaPtr<char> data);
FOR_EACH(SCALE_SPEC, double, float, int, long, long long, unsigned int, unsigned long, unsigned long long)


} /* namespace ddj */
