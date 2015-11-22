/*
 * cuda_array_copy.cu
 *
 *  Created on: Nov 22, 2015
 *      Author: Karol Dzitkowski
 */

#include "cuda_array_copy.hpp"

namespace ddj {

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
			CUDA_ASSERT_RETURN(
				cudaMemcpyAsync(
					result->get()+offset,
					part->get(),
					part->size()*sizeof(T),
					CPY_DTD,
					stream)
			);
			offset += part->size();
		}
	}

	// synchronize streams
	for(int i=0; i < streamCnt; i++)
		CUDA_CALL( cudaStreamSynchronize(streams[i]->Get()) );

	return result;
}


} /* namespace ddj */



