/*
 *  splitter.cu
 *
 *  Created on: 05-10-2015
 *      Author: Karol Dzitkowski
 */

#include "util/other/prefix_sum.cuh"
#include "util/splitter/splitter.hpp"
#include "core/cuda_launcher.cuh"
#include "core/macros.h"

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <tuple>
#include <thrust/sort.h>

namespace ddj
{

template<typename T>
__global__ void _copyIfKernel(
	T* data,
	int* prefixSum_true,
	int* prefixSum_false,
	int* stencil,
    size_t size,
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

// TODO: Change to use transform instead
SharedCudaPtr<int> Splitter::NegateStencil(SharedCudaPtr<int> stencil)
{
	auto result = CudaPtr<int>::make_shared(stencil->size());

	if(stencil->size() > 0)
	{
        this->_policy.setSize(stencil->size());
        cudaLaunch(this->_policy, _negateStencilKernel,
			stencil->get(),
            stencil->size(),
            result->get()
        );
	}
	return result;
}

template<typename T>
SharedCudaPtrTuple<T> Splitter::Split(
	SharedCudaPtr<T> data, SharedCudaPtr<int> stencil)
{
	int out_size = 0;
	auto prefixSum_true = exclusivePrefixSum_thrust(stencil, out_size);
	auto neg_stencil = NegateStencil(stencil);
	auto prefixSum_false = exclusivePrefixSum_thrust(neg_stencil);

	auto yes = CudaPtr<T>::make_shared(out_size);
	auto no = CudaPtr<T>::make_shared(data->size()-out_size);

	if(data->size() > 0)
    {
        this->_policy.setSize(data->size());
        cudaLaunch(this->_policy, _copyIfKernel<T>,
			data->get(),
			prefixSum_true->get(),
			prefixSum_false->get(),
			stencil->get(),
			data->size(),
			yes->get(),
			no->get()
        );
    }

	cudaDeviceSynchronize();

	return std::make_tuple(yes, no);
}

template<typename T>
SharedCudaPtrTuple<T> Splitter::Split2(
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
SharedCudaPtr<T> Splitter::CopyIf(
		SharedCudaPtr<T> data, SharedCudaPtr<int> stencil)
{
	auto result = Split(data, stencil);
	return std::get<0>(result);
}

template<typename T>
SharedCudaPtr<T> Splitter::CopyIfNot(
		SharedCudaPtr<T> data, SharedCudaPtr<int> stencil)
{
	auto result = Split(data, stencil);
	return std::get<1>(result);
}

template<typename T>
__global__ void _mergeKernel(
		const int* stencil,
		const int size,
		const T* in_true,
		const T* in_false,
		const int* prefixSum_true,
		const int* prefixSum_false,
		T* result)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;

	if(stencil[idx])
		result[idx] = in_true[prefixSum_true[idx]];
	else
		result[idx] = in_false[prefixSum_false[idx]];
}

template<typename T>
SharedCudaPtr<T> Splitter::Merge(SharedCudaPtrTuple<T> data, SharedCudaPtr<int> stencil)
{
	// printf("START MERGE KERNEL (data[0] size = %lu, data[1] size = %lu)\n",
	// 	std::get<0>(data)->size(), std::get<1>(data)->size());

	size_t size = stencil->size();
	auto result = CudaPtr<T>::make_shared(size);
	auto prefixSum_true = exclusivePrefixSum_thrust(stencil);
	auto prefixSum_false = exclusivePrefixSum_thrust(NegateStencil(stencil));

	this->_policy.setSize(size);
	cudaLaunch(this->_policy, _mergeKernel<T>,
		stencil->get(),
		size,
		std::get<0>(data)->get(),
		std::get<1>(data)->get(),
		prefixSum_true->get(),
		prefixSum_false->get(),
		result->get()
	);

	cudaDeviceSynchronize();
	CUDA_CALL( cudaGetLastError() );

	return result;
}

#define SPLITTER_SPEC(X) \
	template SharedCudaPtrTuple<X> Splitter::Split<X>(SharedCudaPtr<X>, SharedCudaPtr<int>); \
	template SharedCudaPtrTuple<X> Splitter::Split2<X>(SharedCudaPtr<X>, SharedCudaPtr<int>); \
	template SharedCudaPtr<X> Splitter::CopyIf<X>(SharedCudaPtr<X>, SharedCudaPtr<int>); \
	template SharedCudaPtr<X> Splitter::CopyIfNot<X>(SharedCudaPtr<X>, SharedCudaPtr<int>); \
	template SharedCudaPtr<X> Splitter::Merge<X>(SharedCudaPtrTuple<X>, SharedCudaPtr<int>);
FOR_EACH(SPLITTER_SPEC, char, short, double, float, int, long, long long, unsigned int)


} /* namespace ddj */
