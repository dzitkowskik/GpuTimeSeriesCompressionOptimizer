/*
 * cuda_array_copy.cpp
 *
 *  Created on: Nov 20, 2015
 *      Author: Karol Dzitkowski
 */

#include "util/copy/cuda_array_copy.hpp"

namespace ddj
{

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

#define CUDA_PTR_SPEC(X) \
	template SharedCudaPtr<X> CudaArrayCopy::Concatenate<X>(SharedCudaPtrVector<X>);
FOR_EACH(CUDA_PTR_SPEC, char, double, float, int, long, long long, unsigned int)


} /* namespace ddj */
