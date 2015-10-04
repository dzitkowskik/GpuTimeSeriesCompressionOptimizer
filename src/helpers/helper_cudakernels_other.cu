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

template<typename T, typename UnaryOperator>
__global__ void transformInPlaceKernel(T* data, int size, UnaryOperator op)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	data[idx] = op(data[idx]);
}

template<typename T, typename UnaryOperator>
__global__ void transformKernel(const T* data, int size, UnaryOperator op, T* output)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	output[idx] = op(data[idx]);
}

template<typename T, typename UnaryOperator>
void TransformInPlace(SharedCudaPtr<T> data, UnaryOperator op)
{
	ExecutionPolicy policy;
	policy.setSize(data->size());
	cudaLaunch(policy, transformInPlaceKernel<T, UnaryOperator>,
		data->get(), data->size(), op);
	cudaDeviceSynchronize();
}

template<typename T, typename UnaryOperator>
SharedCudaPtr<T> Transform(SharedCudaPtr<T> data, UnaryOperator op)
{
	ExecutionPolicy policy;
	policy.setSize(data->size());
	auto result = CudaPtr<T>::make_shared(data->size());
	cudaLaunch(policy, transformKernel<T, UnaryOperator>,
		data->get(), data->size(), op, result->get());
	cudaDeviceSynchronize();
	return result;
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
HelperCudaKernels::ModuloKernel(SharedCudaPtr<T> data, T mod)
{
	ModulusOperator<T> op{mod};
    return Transform(data, op);
}

template<typename T> void
HelperCudaKernels::ModuloInPlaceKernel(SharedCudaPtr<T> data, T mod)
{
	ModulusOperator<T> op{mod};
    TransformInPlace(data, op);
}

template<typename T> SharedCudaPtr<T>
HelperCudaKernels::AdditionKernel(SharedCudaPtr<T> data, T val)
{
	AdditionOperator<T> op{val};
    return Transform(data, op);
}

template<typename T> void
HelperCudaKernels::AdditionInPlaceKernel(SharedCudaPtr<T> data, T val)
{
	AdditionOperator<T> op{val};
    TransformInPlace(data, op);
}

template<typename T> SharedCudaPtr<T>
HelperCudaKernels::AbsoluteKernel(SharedCudaPtr<T> data)
{
	AbsoluteOperator<T> op;
    return Transform(data, op);
}

template<typename T> void
HelperCudaKernels::AbsoluteInPlaceKernel(SharedCudaPtr<T> data)
{
	AbsoluteOperator<T> op;
    TransformInPlace(data, op);
}

template<typename T> SharedCudaPtr<T>
HelperCudaKernels::ZeroKernel(SharedCudaPtr<T> data)
{
	ZeroOperator<T> op;
    return Transform(data, op);
}

template<typename T> void
HelperCudaKernels::ZeroInPlaceKernel(SharedCudaPtr<T> data)
{
	ZeroOperator<T> op;
    TransformInPlace(data, op);
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
	template SharedCudaPtr<X> HelperCudaKernels::ZeroKernel<X>(SharedCudaPtr<X>); 			\
	template void HelperCudaKernels::ZeroInPlaceKernel<X>(SharedCudaPtr<X>);				\
	template SharedCudaPtr<X> HelperCudaKernels::AbsoluteKernel<X>(SharedCudaPtr<X>); 		\
	template void HelperCudaKernels::AbsoluteInPlaceKernel<X>(SharedCudaPtr<X>);			\
    template SharedCudaPtr<X> HelperCudaKernels::ModuloKernel<X>(SharedCudaPtr<X>, X); 		\
    template void HelperCudaKernels::ModuloInPlaceKernel<X>(SharedCudaPtr<X>, X);			\
	template SharedCudaPtr<X> HelperCudaKernels::AdditionKernel<X>(SharedCudaPtr<X>, X); 	\
    template void HelperCudaKernels::AdditionInPlaceKernel<X>(SharedCudaPtr<X>, X);
FOR_EACH(MODULO_SPEC, unsigned int, int)

#define CUDA_KERNELS_OTHER_SPEC(X) \
	template SharedCudaPtr<X> HelperCudaKernels::CreateConsecutiveNumbersArray<X>(int, X); \
	template std::tuple<X,X> HelperCudaKernels::MinMax<X>(SharedCudaPtr<X> data);
FOR_EACH(CUDA_KERNELS_OTHER_SPEC, float, int, long int, long long int)

} /* namespace ddj */
