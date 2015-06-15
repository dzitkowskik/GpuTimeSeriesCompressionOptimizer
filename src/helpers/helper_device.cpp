/*
 * helper_cuda.cpp 26-03-2015 Karol Dzitkowski
 */

#include "helper_device.hpp"
#include "helper_macros.h"

namespace ddj {

int HelperDevice::CudaGetDevicesCount()
{
	int count = 0;
	cudaError_t error = cudaGetDeviceCount(&count);
	if(cudaSuccess != error)
		return count;
	else return -1;
}

bool HelperDevice::CudaCheckDeviceForRequirements(int n)
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

cudaError_t HelperDevice::CudaAllocateArray(size_t size, void** array)
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

void HelperDevice::GetMemoryCount(size_t* freeMemory, size_t* totalMemory)
{
	cudaMemGetInfo(freeMemory, totalMemory);
}

int HelperDevice::SetCudaDeviceWithMaxFreeMem()
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
