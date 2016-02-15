/*
 * const_encoding.cu
 *
 *  Created on: 30-10-2015
 *      Author: Karol Dzitkowski
 */

#include "const_encoding.hpp"
#include "util/histogram/histogram.hpp"
#include "util/splitter/splitter.hpp"
#include "util/other/prefix_sum.cuh"
#include "util/other/cuda_array_reduce.cuh"
#include "util/copy/cuda_array_copy.hpp"
#include "core/cuda_launcher.cuh"

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
Stencil ConstEncoding::GetConstStencil(SharedCudaPtr<T> data, T constValue)
{
	Stencil stencil(CudaPtr<int>::make_shared(data->size()));

	this->_policy.setSize(data->size());
	cudaLaunch(this->_policy, _constEncodeStencilKernel<T>,
			data->get(),
			data->size(),
			stencil->get(),
			constValue);
	cudaDeviceSynchronize();

	return stencil;
}

template<typename T>
SharedCudaPtrVector<char> ConstEncoding::Encode(SharedCudaPtr<T> data)
{
	CUDA_ASSERT_RETURN( cudaGetLastError() );
	LOG4CPLUS_INFO_FMT(_logger, "CONST encoding START: data size = %lu", data->size());

	if(data->size() <= 0)
		return SharedCudaPtrVector<char>{
			CudaPtr<char>::make_shared(),
			CudaPtr<char>::make_shared()
		};

	auto mostFrequent = Histogram().GetMostFrequent(data, 1);
	T constValue;
	CUDA_CALL( cudaMemcpy(&constValue, mostFrequent->get(), sizeof(T), CPY_DTH) );

	LOG4CPLUS_TRACE_FMT(_logger,"CONST VALUE = %d\n", (int)constValue);

	auto stencil = GetConstStencil(data, constValue);

	auto resultData = MoveSharedCudaPtr<T, char>(Splitter().CopyIf(data, *stencil));
	auto packedStencil = stencil.pack();
	auto resultMetadata = CudaArrayCopy().Concatenate(
		SharedCudaPtrVector<char> {
			CastSharedCudaPtr<T, char>(mostFrequent),
			packedStencil
		}
	);

	CUDA_ASSERT_RETURN( cudaGetLastError() );
    LOG4CPLUS_INFO(_logger, "CONST enoding END");

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
	LOG4CPLUS_INFO_FMT(
		_logger,
		"CONST decoding START: input[0] size = %lu, input[1] size = %lu",
		input[0]->size(), input[1]->size()
	);

	if(input[0]->size() <= 0) return CudaPtr<T>::make_shared();

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
	
	CUDA_ASSERT_RETURN( cudaGetLastError() );
	LOG4CPLUS_INFO(_logger, "CONST decoding END");

	return result;
}

size_t ConstEncoding::GetCompressedSize(SharedCudaPtr<char> data, DataType type)
{
	if(data->size() <= 0) return 0;
	switch(type)
	{
		case DataType::d_int:
			return GetCompressedSize(boost::reinterpret_pointer_cast<CudaPtr<int>>(data));
		case DataType::d_float:
			return GetCompressedSize(boost::reinterpret_pointer_cast<CudaPtr<float>>(data));
		default:
			throw NotImplementedException("No DictEncoding::GetCompressedSize implementation for that type");
	}
}

template<typename T> size_t ConstEncoding::GetCompressedSize(SharedCudaPtr<T> data)
{
	auto mostFrequent = Histogram().GetMostFrequent(data, 1);
	T constValue;
	CUDA_CALL( cudaMemcpy(&constValue, mostFrequent->get(), sizeof(T), CPY_DTH) );
	auto stencil = GetConstStencil(data, constValue);
	auto notConstCnt = reduce_thrust(*stencil, thrust::plus<int>());
	return notConstCnt * sizeof(T);
}

#define CONST_ENCODING_SPEC(X) \
	template SharedCudaPtrVector<char> ConstEncoding::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> ConstEncoding::Decode<X>(SharedCudaPtrVector<char> data);
FOR_EACH(CONST_ENCODING_SPEC, char, short, double, float, int, long, long long, unsigned int)

} /* namespace ddj */
