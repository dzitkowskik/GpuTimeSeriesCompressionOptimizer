/*
 * cuda_ptr.hpp
 *
 *  Created on: 07-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef CUDA_PTR_H_
#define CUDA_PTR_H_

#include <cuda_runtime_api.h>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/make_shared.hpp>
#include "helpers/helper_macros.h"
#include <vector>
#include <utility>

template<class T> class CudaPtr;

template<class T>
using SharedCudaPtr = boost::shared_ptr<CudaPtr<T>>;

template<class T>
using ScopedCudaPtr = boost::scoped_ptr<CudaPtr<T>>;

template<class T>
using SharedCudaPtrVector = std::vector<SharedCudaPtr<T>>;

template<class T>
using SharedCudaPtrTuple = std::tuple<SharedCudaPtr<T>, SharedCudaPtr<T>>;

template<typename L, typename R>
using SharedCudaPtrPair = std::pair<SharedCudaPtr<L>, SharedCudaPtr<R>>;

template<class T>
class CudaPtr : private boost::noncopyable
{
public:
	CudaPtr() : _pointer(NULL), _size(0){}
	CudaPtr(T* ptr) : _pointer(ptr), _size(0){}
	CudaPtr(size_t size) : _pointer(NULL), _size(size)
	{
		if(_size > 0)
			CUDA_CHECK_RETURN( cudaMalloc((void**)&_pointer, _size*sizeof(T)) );
	}
	CudaPtr(T* ptr, size_t size) : _pointer(ptr), _size(size){}

	~CudaPtr()
	{ CUDA_CHECK_RETURN( cudaFree(_pointer) ); }

public:
	T* get() { return _pointer; }
	size_t size() { return _size; }

	void reset(size_t size)
	{
		_size = size;
		CUDA_CHECK_RETURN( cudaFree(_pointer) );
		CUDA_CHECK_RETURN( cudaMalloc((void**)&_pointer, _size*sizeof(T)) );
	}

	void reset(T* ptr, size_t size = 0)
	{
		CUDA_CHECK_RETURN( cudaFree(_pointer) );
		_pointer = ptr;
		_size = size;
	}

	void fill(T* ptr, size_t size)
	{
		if(_size < size) reset(size);
		CUDA_CHECK_RETURN(
			cudaMemcpy(_pointer, ptr, size*sizeof(T), cudaMemcpyDeviceToDevice)
			);
	}

	void fillFromHost(T* ptr, size_t size)
	{
		if(_size < size) reset(size);
		CUDA_CHECK_RETURN(
			cudaMemcpy(_pointer, ptr, size*sizeof(T), cudaMemcpyHostToDevice)
			);
	}

	SharedCudaPtr<T> copy()
	{
		SharedCudaPtr<T> result;
		result.fill(_pointer, _size);
		return result;
	}

public:
	static SharedCudaPtr<T> make_shared()
	{ return SharedCudaPtr<T>(new CudaPtr()); }
	static SharedCudaPtr<T> make_shared(size_t size)
	{ return SharedCudaPtr<T>(new CudaPtr(size)); }
	static SharedCudaPtr<T> make_shared(T* ptr)
	{ return SharedCudaPtr<T>(new CudaPtr(ptr)); }
	static SharedCudaPtr<T> make_shared(T* ptr, size_t size)
	{ return SharedCudaPtr<T>(new CudaPtr(ptr, size)); }

private:
	T* _pointer;
	size_t _size;
};

#endif /* CUDA_PTR_H_ */
