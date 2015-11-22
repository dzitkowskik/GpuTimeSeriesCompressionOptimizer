/*
 * cuda_stream.hpp
 *
 *  Created on: Nov 20, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef CUDA_STREAM_HPP_
#define CUDA_STREAM_HPP_

#include "core/macros.h"

#include <cuda_runtime_api.h>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>

class CudaStream;

using SharedCudaStream = boost::shared_ptr<CudaStream>;
using ScopedCudaStream = boost::scoped_ptr<CudaStream>;
using SharedCudaStreamVector = std::vector<SharedCudaStream>;

class CudaStream : private boost::noncopyable
{
public:
	CudaStream()
	{ CUDA_ASSERT_RETURN( cudaStreamCreate(&_stream) ); }
	~CudaStream()
	{ CUDA_ASSERT_RETURN( cudaStreamDestroy(_stream) ); }

public:
	cudaStream_t Get() { return _stream; }

public:
	static SharedCudaStream make_shared()
	{ return SharedCudaStream(new CudaStream()); }

	static SharedCudaStreamVector make_shared(size_t size)
	{
		std::vector<SharedCudaStream> result;
		for(int i = 0; i < size; i++)
			result.push_back(SharedCudaStream(new CudaStream()));
		return result;
	}

private:
	cudaStream_t _stream;
};


#endif /* CUDA_STREAM_HPP_ */
