#include "scale.cuh"
#include "helpers/helper_macros.h"
#include <cuda_runtime_api.h>
#include "compression/macros.cuh"

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

#define SCALE_SPEC(X) \
	template X* scaleEncode<X>(X* data, int size, X& min); \
	template X* scaleDecode<X>(X* data, int size, X& min);
FOR_EACH(SCALE_SPEC, double, float, int, long, long long, unsigned int, unsigned long, unsigned long long)

} /* namespace ddj */
