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
SharedCudaPtr<char> scaleEncode(SharedCudaPtr<T> data, T& min)
{
	int block_size = SCALE_ENCODING_GPU_BLOCK_SIZE;
	int block_cnt = (data->size() + block_size - 1) / block_size;

	auto result = CudaPtr<char>::make_shared(data->size()*sizeof(T));

	scaleEncodeKernel<T><<<block_size, block_cnt>>>(
			data->get(),
			data->size(),
			(T*)result->get(),
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
SharedCudaPtr<T> scaleDecode(SharedCudaPtr<char> data, T& min)
{
	int block_size = SCALE_ENCODING_GPU_BLOCK_SIZE;
	int block_cnt = (data->size() + block_size - 1) / block_size;

	auto result = CudaPtr<T>::make_shared(data->size());

	scaleDecodeKernel<T><<<block_size, block_cnt>>>(
			(T*)data->get(),
			data->size(),
			result->get(),
			min);

	cudaDeviceSynchronize();

	return result;
}

#define SCALE_SPEC(X) \
	template SharedCudaPtr<char> scaleEncode<X>(SharedCudaPtr<X> data, X& min); \
	template SharedCudaPtr<X> scaleDecode<X>(SharedCudaPtr<char> data, X& min);
FOR_EACH(SCALE_SPEC, double, float, int, long, long long, unsigned int, unsigned long, unsigned long long)

} /* namespace ddj */
