/*
 * helper_device.hpp 26-03-2015 Karol Dzitkowski
 */

#ifndef DDJ_HELPER_DEVICE_H_
#define DDJ_HELPER_DEVICE_H_

#include "core/config.hpp"
#include <cuda_runtime_api.h>

namespace ddj {

class HelperDevice {
private:
	Config* _config;

public:
	HelperDevice()
        : _config(Config::GetInstance()) { }

	int    CudaGetDevicesCount();
	bool   CudaCheckDeviceForRequirements(int n);
	void   GetMemoryCount(size_t* freeMemory, size_t* totalMemory);
	int    SetCudaDeviceWithMaxFreeMem();
    cudaError_t CudaAllocateArray(size_t size, void** array);
};

} /* namespace ddj */
#endif /* DDJ_HELPER_DEVICE_H_ */
