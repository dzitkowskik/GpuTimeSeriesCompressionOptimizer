/*
 * helper_cudakernels_other.cu
 *
 *  Created on: 09-04-2015
 *      Author: Karol Dzitkowski
 */

#include "helper_cudakernels.cuh"
#include "core/operators.cuh"
#include "core/execution_policy.hpp"
#include "helpers/helper_cuda.cuh"
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>

#define MODULO_KERNEL_BLOCK_SIZE 32
#define CREATE_NUM_KERNEL_BLOCK_SIZE 32

namespace ddj
{

template<typename T>
__global__ void _createConsecutiveNumbersArrayKernel(
    T* data, int size, T start)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	data[idx] = start + idx;
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

#define CUDA_KERNELS_OTHER_SPEC(X) \
	template SharedCudaPtr<X> HelperCudaKernels::CreateConsecutiveNumbersArray<X>(int, X); \
	template std::tuple<X,X> HelperCudaKernels::MinMax<X>(SharedCudaPtr<X> data);
FOR_EACH(CUDA_KERNELS_OTHER_SPEC, float, int, long int, long long int)

} /* namespace ddj */
