#include "delta.cuh"
#include "helpers/helper_macros.h"
#include <cuda_runtime_api.h>

//a  a  b  b  a  a  b  b  a  a  b  b  a  a
//1  2  2  3  3  4  4  4  5  1  2  3  3  3
// 1  0  1  0  1  0  0  1  -4 1  1  0  0

#define DELTA_ENCODING_GPU_BLOCK_SIZE 64
#define DELTA_DECODING_GPU_BLOCK_SIZE 64

namespace ddj
{

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
T* deltaEncode(T* data, int size, T& first)
{
	int result_size = size - 1;
	int block_size = DELTA_ENCODING_GPU_BLOCK_SIZE;
	int block_cnt = (size + block_size - 1) / block_size;

	T* result;
	CUDA_CALL( cudaMalloc((void**)&result, result_size*sizeof(T)) );
	deltaEncodeKernel<T><<<block_size, block_cnt>>>(data, size, result);
	CUDA_CALL( cudaMemcpy(&first, data, sizeof(T), cudaMemcpyDeviceToHost) );
	cudaDeviceSynchronize();

	return result;
}

template<typename T>
void deltaEncodeInPlace(T* data, int size)
{
	int block_size = DELTA_ENCODING_GPU_BLOCK_SIZE;
	int block_cnt = (size + block_size - 1) / block_size;

	deltaEncodeKernel<T><<<block_size, block_cnt>>>(data, size, data+1);
	cudaDeviceSynchronize();
}

template<typename T>
__global__ void deltaDecodeKernel(T* data, int size, T first, T* result)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= size) return;
	for(int i=0; i<idx; i++)
		first += data[i];
	result[idx] = first;
}

template<typename T>
T* deltaDecode(T* data, int size, T first)
{
	int block_size = DELTA_DECODING_GPU_BLOCK_SIZE;
	int block_cnt = (size + block_size - 1) / block_size;

	T* result;
	CUDA_CALL( cudaMalloc((void**)&result, size*sizeof(T)) );
	deltaDecodeKernel<T><<<block_size, block_cnt>>>(data, size, first, result);
	cudaDeviceSynchronize();

	return result;
}

template<typename T>
void deltaDecodeInPlace(T* data, int size)
{
	int block_size = DELTA_DECODING_GPU_BLOCK_SIZE;
	int block_cnt = (size + block_size - 1) / block_size;

	T first;
	T* result;
	CUDA_CALL( cudaMemcpy(&first, data, sizeof(T), cudaMemcpyDeviceToHost) );
	CUDA_CALL( cudaMalloc((void**)result, size*sizeof(T)) );
	deltaDecodeKernel<T><<<block_size, block_cnt>>>(data+1, size, first, result);
	cudaDeviceSynchronize();
	CUDA_CALL( cudaMemcpy(data, result, size*sizeof(T), cudaMemcpyDeviceToDevice) );
	CUDA_CALL( cudaFree(result) );
}

template float* deltaEncode<float>(float* data, int size, float& first);
template void deltaEncodeInPlace<float>(float* data, int size);
template float* deltaDecode<float>(float* data, int size, float first);
template void deltaDecodeInPlace<float>(float* data, int size);

} /* namespace ddj */
