/*
 * helper_cudakernels.cu
 *
 *  Created on: 09-04-2015
 *      Author: Karol Dzitkowski
 */

#include "utils/prefix_sum.cuh"
#include "helper_cudakernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <tuple>

#define SPLIT_KERNEL_BLOCK_SIZE 32

namespace ddj
{

template<typename T>
__global__ void _copyIfKernel(
	T* data, int* prefixSum, int* stencil, size_t size, T* out_true, T* out_false)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	size_t out_idx = prefixSum[idx];
	T value = data[idx];
	if (stencil[idx]) out_true[out_idx] = value;
	else out_false[out_idx] = value;
}

template<typename T>
std::tuple<SharedCudaPtr<T>> SplitKernel(
	SharedCudaPtr<T> data, SharedCudaPtr<int> stencil)
{
	int out_size = 0;
	int block_size = SPLIT_KERNEL_BLOCK_SIZE;
	int block_cnt = (data->size() + block_size - 1) / block_size;
	auto prefixSum = inclusivePrefixSum_thrust(stencil, out_size);
	auto yes = CudaPtr<T>::make_shared(out_size);
	auto no = CudaPtr<T>::make_shared(data->size()-out_size);

	_copyIfKernel<<<block_size, block_cnt>>>(
		data->get(),
		prefixSum->get(),
		stencil->get(),
		data->size(),
		yes->get(),
		no->get());

	return std::make_tuple(yes, no);
}

template<typename T>
SharedCudaPtr<T> CopyIfKernel(SharedCudaPtr<T> data, SharedCudaPtr<int> stencil)
{
	auto result = SplitKernel(data, stencil);
	return std::get<0>(result);
}

template<typename T>
SharedCudaPtr<T> CopyIfNotKernel(SharedCudaPtr<T> data, SharedCudaPtr<int> stencil)
{
	auto result = SplitKernel(data, stencil);
	return std::get<1>(result);
}

} /* namespace ddj */
