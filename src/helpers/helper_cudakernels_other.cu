/*
 * helper_cudakernels_other.cu
 *
 *  Created on: 09-04-2015
 *      Author: Karol Dzitkowski
 */

#include "helper_cudakernels.cuh"
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>

#define MODULO_KERNEL_BLOCK_SIZE 32
#define CREATE_NUM_KERNEL_BLOCK_SIZE 32

namespace ddj
{

template<typename T>
__global__ void _moduloInPlaceKernel(T* data, int size, int mod)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	data[idx] %= mod;
}

template<typename T>
__global__ void _moduloKernel(T* data, int size, int mod, T* out)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	out[idx] = data[idx] % mod;
}

template<typename T>
__global__ void _createConsecutiveNumbersArrayKernel(
    T* data, int size, T start)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	data[idx] = start + idx;
}

template<typename T> SharedCudaPtr<T>
HelperCudaKernels::ModuloKernel(SharedCudaPtr<T> data, int mod)
{
    const int tpb = MODULO_KERNEL_BLOCK_SIZE;
    int blocks = (data->size() + tpb - 1) / tpb;
    auto result = CudaPtr<T>::make_shared(data->size());
    _moduloKernel<<<blocks, tpb>>>(
        data->get(), data->size(), mod, result->get());
    return result;
}

template<typename T> void
HelperCudaKernels::ModuloInPlaceKernel(SharedCudaPtr<T> data, int mod)
{
    const int tpb = MODULO_KERNEL_BLOCK_SIZE;
    int blocks = (data->size() + tpb - 1) / tpb;
    _moduloInPlaceKernel<<<blocks, tpb>>>(data->get(), data->size(), mod);
}

template<typename T> SharedCudaPtr<T>
HelperCudaKernels::CreateConsecutiveNumbersArray(int size, T start)
{
    const int tpb = CREATE_NUM_KERNEL_BLOCK_SIZE;
    int blocks = (size + tpb - 1) / tpb;
    auto result = CudaPtr<T>::make_shared(size);
    _createConsecutiveNumbersArrayKernel<<<blocks, tpb>>>(
        result->get(), size, start);
    return result;
}

template<typename T> std::tuple<T,T> HelperCudaKernels::MinMax(SharedCudaPtr<T> data)
{
	thrust::device_ptr<T> dp(data->get());
	auto tuple = thrust::minmax_element(dp, dp+data->size());
	T min = *(tuple.first);
	T max = *(tuple.second);
	return std::make_tuple(min, max);
}

#define MODULO_SPEC(X) \
    template SharedCudaPtr<X> HelperCudaKernels::ModuloKernel<X>(SharedCudaPtr<X>, int); \
    template void HelperCudaKernels::ModuloInPlaceKernel<X>(SharedCudaPtr<X>, int);
FOR_EACH(MODULO_SPEC, unsigned int, int)

#define CUDA_KERNELS_OTHER_SPEC(X) \
	template SharedCudaPtr<X> HelperCudaKernels::CreateConsecutiveNumbersArray<X>(int, X); \
	template std::tuple<X,X> HelperCudaKernels::MinMax<X>(SharedCudaPtr<X> data);
FOR_EACH(CUDA_KERNELS_OTHER_SPEC, float, int, long int, long long int)

} /* namespace ddj */
