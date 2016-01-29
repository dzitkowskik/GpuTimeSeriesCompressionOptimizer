/*
 * cuda_cuda.cpp 26-03-2015 Karol Dzitkowski
 */

#include "core/cuda_device.hpp"
#include "core/macros.h"

namespace ddj {

int CudaDevice::CudaGetDevicesCount()
{
	int count = 0;
	cudaError_t error = cudaGetDeviceCount(&count);
	if(cudaSuccess == error)
		return count;
	// ELSE
	CUDA_CALL(error);
	return 0;
}

bool CudaDevice::CudaCheckDeviceForRequirements(int n)
{
	int driverVersion = 0, runtimeVersion = 0;
	cudaDeviceProp prop;
	cudaSetDevice(n);
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	cudaGetDeviceProperties(&prop, n);

	if(prop.major < _config->GetValue<int>(std::string("MIN_CUDA_MAJOR_VER")))
        return false;
	if(prop.minor < _config->GetValue<int>(std::string("MIN_CUDA_MINOR_VER")))
        return false;

	return true;
}

cudaError_t CudaDevice::CudaAllocateArray(size_t size, void** array)
{
	size_t mbSize = this->_config->GetValue<int>(std::string("MB_SIZE_IN_BYTES"));
	size_t freeMemory, totalMemory;
	cudaMemGetInfo(&freeMemory, &totalMemory);

	cudaError_t result = cudaSuccess;
	if(totalMemory <= size)
	{
		result = cudaErrorMemoryAllocation;
		return result;
	}
	result = cudaMalloc((void**)array, size);
	cudaMemGetInfo(&freeMemory, &totalMemory);

	return result;
}

void CudaDevice::GetMemoryCount(size_t* freeMemory, size_t* totalMemory)
{
	cudaMemGetInfo(freeMemory, totalMemory);
}

int CudaDevice::SetCudaDeviceWithMaxFreeMem()
{
	int deviceId = 0;
	size_t free;
	size_t total;
	size_t max_free = 0;
	int devCount = CudaGetDevicesCount();
	for(int i=0; i<devCount; i++)
	{
		cudaSetDevice(i);
		GetMemoryCount(&free, &total);
//		printf("Device %d has %d free memory\n", i, (int)free);
		if(free > max_free)
		{
			max_free = free;
			deviceId = i;
		}
	}
	cudaSetDevice(deviceId);
	return deviceId;
}

} /* namespace ddj */
