/*
 * helper_cudakernels_split.cu
 *
 *  Created on: 09-04-2015
 *      Author: Karol Dzitkowski
 */

#include "utils/prefix_sum.cuh"
#include "helper_cudakernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <tuple>
#include <thrust/sort.h>

#define SPLIT_KERNEL_BLOCK_SIZE 32
#define NEGATE_KERNEL_BLOCK_SIZE 32

namespace ddj
{

template<typename T>
__global__ void _copyIfKernel(
	T* data,
	int* prefixSum_true,
	int* prefixSum_false,
	int* stencil, size_t size,
	T* out_true,
	T* out_false)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;

	T value = data[idx];
	if (stencil[idx])
		out_true[prefixSum_true[idx]] = value;
	else
		out_false[prefixSum_false[idx]] = value;
}

__global__ void _negateStencilKernel(int* stencil, int size, int* out)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;

	out[idx] = stencil[idx] == 1 ? 0 : 1;
}

SharedCudaPtr<int> NegateStencilKernel(SharedCudaPtr<int> stencil)
{
	int block_size = NEGATE_KERNEL_BLOCK_SIZE;
	int block_cnt = (stencil->size() + block_size - 1) / block_size;

	auto result = CudaPtr<int>::make_shared(stencil->size());

	_negateStencilKernel<<<block_size, block_cnt>>>(
			stencil->get(), stencil->size(), result->get());

	return result;
}

template<typename T>
SharedCudaPtrTuple<T> HelperCudaKernels::SplitKernel(
	SharedCudaPtr<T> data, SharedCudaPtr<int> stencil)
{
	int out_size = 0;
	int block_size = SPLIT_KERNEL_BLOCK_SIZE;
	int block_cnt = (data->size() + block_size - 1) / block_size;
	auto prefixSum_true = exclusivePrefixSum_thrust(stencil, out_size);
	auto neg_stencil = NegateStencilKernel(stencil);
	auto prefixSum_false = exclusivePrefixSum_thrust(neg_stencil);

	auto yes = CudaPtr<T>::make_shared(out_size);
	auto no = CudaPtr<T>::make_shared(data->size()-out_size);

	if(data->size() > 0)
		_copyIfKernel<<<block_size, block_cnt>>>(
			data->get(),
			prefixSum_true->get(),
			prefixSum_false->get(),
			stencil->get(),
			data->size(),
			yes->get(),
			no->get());

	return std::make_tuple(yes, no);
}

template<typename T>
SharedCudaPtrTuple<T> HelperCudaKernels::SplitKernel2(
		SharedCudaPtr<T> data, SharedCudaPtr<int> stencil)
{
	int size = data->size();
	thrust::device_ptr<T> data_ptr(data->get());
	thrust::device_ptr<int> stencil_ptr(stencil->get());
	thrust::stable_sort_by_key(stencil_ptr, stencil_ptr+size, data_ptr);

	int yes_size = thrust::count(stencil_ptr, stencil_ptr + size, 1);
	int no_size = size - yes_size;

	auto no = CudaPtr<T>::make_shared(no_size);
	auto yes = CudaPtr<T>::make_shared(yes_size);

	no->fill(data->get(), no_size);
	yes->fill(data->get()+no_size, yes_size);

	return std::make_tuple(yes, no);
}

template<typename T>
SharedCudaPtr<T> HelperCudaKernels::CopyIfKernel(
		SharedCudaPtr<T> data, SharedCudaPtr<int> stencil)
{
	auto result = SplitKernel(data, stencil);
	return std::get<0>(result);
}

template<typename T>
SharedCudaPtr<T> HelperCudaKernels::CopyIfNotKernel(
		SharedCudaPtr<T> data, SharedCudaPtr<int> stencil)
{
	auto result = SplitKernel(data, stencil);
	return std::get<1>(result);
}

#define CUDA_KERNELS_SPEC(X) \
	template SharedCudaPtrTuple<X> \
	HelperCudaKernels::SplitKernel<X>(SharedCudaPtr<X>, SharedCudaPtr<int>); \
	template SharedCudaPtrTuple<X> \
	HelperCudaKernels::SplitKernel2<X>(SharedCudaPtr<X>, SharedCudaPtr<int>); \
	template SharedCudaPtr<X> \
	HelperCudaKernels::CopyIfKernel<X>(SharedCudaPtr<X>, SharedCudaPtr<int>); \
	template SharedCudaPtr<X> \
	HelperCudaKernels::CopyIfNotKernel<X>(SharedCudaPtr<X>, SharedCudaPtr<int>);
FOR_EACH(CUDA_KERNELS_SPEC, float, int)


} /* namespace ddj */
