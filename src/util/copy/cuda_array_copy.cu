/*
 * cuda_array_copy.cu
 *
 *  Created on: Nov 22, 2015
 *      Author: Karol Dzitkowski
 */

#include "cuda_array_copy.hpp"
#include "core/cuda_launcher.cuh"

namespace ddj {

template<typename T>
SharedCudaPtr<T> CudaArrayCopy::Concatenate(SharedCudaPtrVector<T> data)
{
	size_t totalSize = 0, i = 0, offset = 0;
	for(auto& part : data)
		totalSize += part->size();
	auto result = CudaPtr<T>::make_shared(totalSize);

	for(auto& part : data)
	{
		if(part->size() > 0)
		{
			CUDA_ASSERT_RETURN(
				cudaMemcpy(result->get()+offset, part->get(), part->size()*sizeof(T), CPY_DTD)
			);
			offset += part->size();
		}
	}

	return result;
}

template<typename T>
__global__ void _copyKernel(T* source, size_t size, T* destination)
{
	unsigned long int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= size) return;
	destination[idx] = source[idx];
}

template<typename T>
SharedCudaPtr<T> CudaArrayCopy::ConcatenateParallel(SharedCudaPtrVector<T> data, SharedCudaStreamVector streams)
{
	int streamCnt = streams.size();
	size_t totalSize = 0, offset = 0;
	cudaStream_t stream;

	// get total data parts count and prepare so many streams
	for(auto& part : data)
		totalSize += part->size();

	auto result = CudaPtr<T>::make_shared(totalSize);

	// copy each part to output with offset in different stream
	for(int i = 0; i < data.size(); i++)
	{
		auto part = data[i];
		stream = streams[i%streamCnt]->Get();
		if(part->size() > 0)
		{
			this->_policy.setSize(part->size());
			this->_policy.setStream(stream);
			cudaLaunch(this->_policy, _copyKernel<T>,
					part->get(),
					part->size(),
					result->get()+offset);

			offset += part->size();
		}
	}

	// synchronize streams
	for(int i=0; i < streamCnt; i++)
		CUDA_CALL( cudaStreamSynchronize(streams[i]->Get()) );

	return result;
}

#define CONCATENATE_SPEC(X) \
	template SharedCudaPtr<X> CudaArrayCopy::Concatenate<X>(SharedCudaPtrVector<X>); \
	template SharedCudaPtr<X> CudaArrayCopy::ConcatenateParallel<X>(SharedCudaPtrVector<X>, SharedCudaStreamVector);
FOR_EACH(CONCATENATE_SPEC, char, double, float, int, long, long long, unsigned int)



} /* namespace ddj */



