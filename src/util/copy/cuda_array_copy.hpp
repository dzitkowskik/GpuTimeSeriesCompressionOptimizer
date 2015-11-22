/*
 * cuda_array_copy.hpp
 *
 *  Created on: Nov 20, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef CUDA_ARRAY_COPY_HPP_
#define CUDA_ARRAY_COPY_HPP_

#include "core/cuda_ptr.hpp"
#include "core/cuda_stream.hpp"

namespace ddj
{

class CudaArrayCopy
{
public:
	CudaArrayCopy() {}
	virtual ~CudaArrayCopy() {}

public:
	template<typename T>
	SharedCudaPtr<T> Concatenate(SharedCudaPtrVector<T> data);

	template<typename T>
	SharedCudaPtr<T> ConcatenateParallel(SharedCudaPtrVector<T> data, SharedCudaStreamVector streams);
};

} /* namespace ddj */

#endif /* CUDA_ARRAY_COPY_HPP_ */
