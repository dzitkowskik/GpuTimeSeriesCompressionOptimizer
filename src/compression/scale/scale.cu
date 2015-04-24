#include "scale.cuh"
#include "helpers/helper_macros.h"
#include <cuda_runtime_api.h>

#define SCALE_ENCODING_GPU_BLOCK_SIZE 64
#define SCALE_DECODING_GPU_BLOCK_SIZE 64

namespace ddj
{

template<typename T>
__global__ void scaleEncodeKernel(T* data, int size, T* result, T min)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= size) return;
	result[idx] = data[idx] - min;
}

template<typename T>
T* scaleEncode(T* data, int size, T& min)
{
	int block_size = SCALE_ENCODING_GPU_BLOCK_SIZE;
	int block_cnt = (size + block_size - 1) / block_size;

	T* result;
	CUDA_CALL( cudaMalloc((void**)&result, size*sizeof(T)) );
	scaleEncodeKernel<T><<<block_size, block_cnt>>>(data, size, result, min);
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
T* scaleDecode(T* data, int size, T& min)
{
	int block_size = SCALE_ENCODING_GPU_BLOCK_SIZE;
	int block_cnt = (size + block_size - 1) / block_size;

	T* result;
	CUDA_CALL( cudaMalloc((void**)&result, size*sizeof(T)) );
	scaleDecodeKernel<T><<<block_size, block_cnt>>>(data, size, result, min);
	cudaDeviceSynchronize();

	return result;
}

template float* scaleEncode<float>(float* data, int size, float& min);
template float* scaleDecode<float>(float* data, int size, float& min);

} /* namespace ddj */
