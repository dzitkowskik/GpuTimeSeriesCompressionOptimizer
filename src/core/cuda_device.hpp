/*
 * cuda_device.hpp 26-03-2015 Karol Dzitkowski
 */

#ifndef DDJ_CORE_CUDA_DEVICE_HPP_
#define DDJ_CORE_CUDA_DEVICE_HPP_

#include "core/config.hpp"
#include <cuda_runtime_api.h>

namespace ddj {

class CudaDevice {
private:
	Config* _config;

public:
	CudaDevice()
        : _config(Config::GetInstance()) { }

	int    CudaGetDevicesCount();
	bool   CudaCheckDeviceForRequirements(int n);
	void   GetMemoryCount(size_t* freeMemory, size_t* totalMemory);
	int    SetCudaDeviceWithMaxFreeMem();
    cudaError_t CudaAllocateArray(size_t size, void** array);
};

} /* namespace ddj */
#endif /* DDJ_CORE_CUDA_DEVICE_HPP_ */
