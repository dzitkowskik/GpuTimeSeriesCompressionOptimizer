/*
 * cuda_ptr.hpp
 *
 *  Created on: 07-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef CUDA_PTR_H_
#define CUDA_PTR_H_

#include "core/macros.h"

#include <cuda_runtime_api.h>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/make_shared.hpp>
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
	CudaPtr(size_t size) : _pointer(NULL), _size(size*sizeof(T))
	{
		if(_size > 0)
			CUDA_ASSERT_RETURN( cudaMalloc((void**)&_pointer, _size) );
	}
	CudaPtr(T* pointer, size_t size) : _pointer(pointer), _size(size*sizeof(T)) {}

	~CudaPtr()
	{
		if(_pointer != nullptr)
			CUDA_ASSERT_RETURN( cudaFree(_pointer) );
	}

public:
	T* get() { return _pointer; }
	size_t size() { return _size/sizeof(T); }

	void reset(size_t size)
	{
		if(this->size() >= size) return;
		_size = size*sizeof(T);
		CUDA_ASSERT_RETURN( cudaFree(_pointer) );
		CUDA_ASSERT_RETURN( cudaMalloc((void**)&_pointer, _size) );
	}

	void reset(T* ptr, size_t size = 0)
	{
		CUDA_ASSERT_RETURN( cudaFree(_pointer) );
		_pointer = ptr;
		_size = size*sizeof(T);
	}

	void clear() { reset(nullptr, 0); }

	void fill(T* ptr, size_t size)
	{
		reset(size);
		CUDA_ASSERT_RETURN(
			cudaMemcpy(_pointer, ptr, size*sizeof(T), cudaMemcpyDeviceToDevice)
			);
	}

	void fillFromHost(const T* ptr, size_t size)
	{
		reset(size);
		CUDA_ASSERT_RETURN(
			cudaMemcpy(_pointer, ptr, size*sizeof(T), cudaMemcpyHostToDevice)
			);
	}

	void set(int value)
	{
		CUDA_ASSERT_RETURN(
				cudaMemset(_pointer, value, _size);
				;)
	}

	SharedCudaPtr<T> copy()
	{
		auto result = CudaPtr<char>::make_shared(_size);
		result->fill((char*)_pointer, _size);
		return boost::reinterpret_pointer_cast<CudaPtr<T>>(result);
	}

	SharedCudaPtr<T> copy(size_t size)
	{
		auto result = make_shared(size);
		result->fill(_pointer, size);
		return result;
	}

	boost::shared_ptr<std::vector<T>> copyToHost()
	{
		boost::shared_ptr<std::vector<T>> result(new std::vector<T>(size()));
		CUDA_ASSERT_RETURN(
			cudaMemcpy( result->data(), _pointer, _size, CPY_DTH )
			);
		return result;
	}

	template<typename FROM, typename TO>
	friend SharedCudaPtr<TO> MoveSharedCudaPtr(SharedCudaPtr<FROM> data);

private:
	template<typename S> CudaPtr<S>* move()
	{
		size_t new_size = _size / sizeof(S);
		auto result = new CudaPtr<S>((S*)(this->_pointer), new_size);
		this->_pointer = 0;
		this->_size = 0;
		return result;
	}

public:
	static SharedCudaPtr<T> make_shared()
	{ return SharedCudaPtr<T>(new CudaPtr()); }
	static SharedCudaPtr<T> make_shared(size_t size)
	{ return SharedCudaPtr<T>(new CudaPtr(size)); }
	static SharedCudaPtr<T> make_shared(T* data, size_t size)
	{ return SharedCudaPtr<T>(new CudaPtr(data, size)); }

private:
	T* _pointer;
	size_t _size;
};

template<typename FROM, typename TO>
SharedCudaPtr<TO> MoveSharedCudaPtr(SharedCudaPtr<FROM> data)
{
	return SharedCudaPtr<TO>(data->template move<TO>());
}

template<typename FROM, typename TO>
SharedCudaPtr<TO> CastSharedCudaPtr(SharedCudaPtr<FROM> data)
{
	return boost::reinterpret_pointer_cast<CudaPtr<TO>>(data);
}

#endif /* CUDA_PTR_H_ */
