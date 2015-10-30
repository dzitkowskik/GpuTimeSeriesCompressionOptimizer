/*
 * const_encoding.cu
 *
 *  Created on: 30-10-2015
 *      Author: Karol Dzitkowski
 */

#include "const_encoding.hpp"
#include "util/histogram/histogram.hpp"
#include "util/splitter/splitter.hpp"
#include "util/stencil/stencil.hpp"
#include "util/other/prefix_sum.cuh"
#include "helpers/helper_cuda.cuh"

namespace ddj
{

template<typename T>
__global__ void _constEncodeStencilKernel(T* data, size_t size, int* stencil, T constValue)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= size) return;
	stencil[idx] = data[idx] != constValue;
}

template<typename T>
SharedCudaPtrVector<char> ConstEncoding::Encode(SharedCudaPtr<T> data)
{
	auto mostFrequent = Histogram().GetMostFrequent(data, 1);
	T constValue;
	CUDA_CALL( cudaMemcpy(&constValue, mostFrequent->get(), sizeof(T), CPY_DTH) );
	Stencil stencil(CudaPtr<int>::make_shared(data->size()));

	this->_policy.setSize(data->size());
	cudaLaunch(this->_policy, _constEncodeStencilKernel<T>,
			data->get(),
			data->size(),
			stencil->get(),
			constValue);
	cudaDeviceSynchronize();

	auto resultData = MoveSharedCudaPtr<T, char>(Splitter().CopyIf(data, *stencil));
	auto packedStencil = stencil.pack();
	auto resultMetadata = Concatenate(
		SharedCudaPtrVector<char> {
			MoveSharedCudaPtr<T, char>(mostFrequent),
			packedStencil
		}
	);

	return SharedCudaPtrVector<char> { resultMetadata, resultData };
}

template<typename T>
__global__ void _constDecodeKernel(
		T constValue,
		int* stencil,
		int* indexes,
		size_t size,
		T* otherData,
		T* output)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= size) return;
	if(stencil[idx])
		output[idx] = otherData[indexes[idx]];
	else
		output[idx] = constValue;
}

template<typename T>
SharedCudaPtr<T> ConstEncoding::Decode(SharedCudaPtrVector<char> input)
{
	auto metadata = input[0];
	auto data = input[1];

	T constValue;
	CUDA_CALL( cudaMemcpy(&constValue, metadata->get(), sizeof(T), CPY_DTH) );

	Stencil stencil(metadata, sizeof(T));

	auto indexes = exclusivePrefixSum_thrust(*stencil);
	auto result = CudaPtr<T>::make_shared(stencil->size());

	this->_policy.setSize(stencil->size());
	cudaLaunch(this->_policy, _constDecodeKernel<T>,
			constValue,
			stencil->get(),
			indexes->get(),
			stencil->size(),
			(T*)data->get(),
			result->get());
	cudaDeviceSynchronize();

	return result;
}

#define CONST_ENCODING_SPEC(X) \
	template SharedCudaPtrVector<char> ConstEncoding::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> ConstEncoding::Decode<X>(SharedCudaPtrVector<char> data);
FOR_EACH(CONST_ENCODING_SPEC, float, int, long long, unsigned int)

} /* namespace ddj */
