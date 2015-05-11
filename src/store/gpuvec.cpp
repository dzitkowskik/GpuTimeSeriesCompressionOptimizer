#include "gpuvec.hpp"
#include <cuda_runtime_api.h>
#include "helpers/helper_macros.h"

namespace ddj
{

GpuVec::GpuVec() :
		_logger(Logger::getRoot()), _config(Config::GetInstance())
{ Init(); }

GpuVec::GpuVec(ullint size) :
		_logger(Logger::getRoot()), _config(Config::GetInstance())
{
	Init();
	Resize(size);
}

GpuVec::~GpuVec()
{
	CUDA_CALL(cudaFree(_memoryPointer));
}

void GpuVec::Init()
{
	_memoryCapacity = 0;
	_memoryOffset = 0;
	_memoryPointer = nullptr;
}

ullint GpuVec::Size()
{
	return _memoryCapacity;
}

ullint GpuVec::Write(void* data, ullint size)
{
	boost::mutex::scoped_lock lock(_offsetMutex);

	while(_memoryOffset+size > _memoryCapacity)
		this->Resize(2*_memoryCapacity+1);

	CUDA_CALL(cudaMemcpy((char*)_memoryPointer+_memoryOffset, data, size, cudaMemcpyDeviceToDevice));
	ullint dataOffset = _memoryOffset;
	_memoryOffset += size;
	return dataOffset;
}

void* GpuVec::Read(ullint offset, ullint size)
{
	boost::mutex::scoped_lock lock(_offsetMutex);
	return (char*)_memoryPointer+offset;
}

void* GpuVec::Get(ullint offset, ullint size)
{
	boost::mutex::scoped_lock lock(_offsetMutex);
	void* tmp;
	CUDA_CALL(cudaMalloc((void** )&tmp, size));
	CUDA_CALL(cudaMemcpy(tmp, (char*)_memoryPointer+offset, size, cudaMemcpyDeviceToDevice));
	return tmp;
}

void GpuVec::Resize(ullint size)
{
	if (size == _memoryCapacity) return;
	void* tmp;
	CUDA_CALL(cudaMalloc((void** )&tmp, size));
	CUDA_CALL(cudaMemset(tmp, 0, size));
	if (_memoryCapacity > 0)
	{
		if (size >= _memoryCapacity)
			CUDA_CALL(cudaMemcpy(tmp, _memoryPointer, _memoryCapacity, cudaMemcpyDeviceToDevice));
		else
			CUDA_CALL(cudaMemcpy(tmp, _memoryPointer, size, cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaFree(_memoryPointer));
	}
	_memoryPointer = tmp;
	_memoryCapacity = size;
}

} /* namespace ddj */
