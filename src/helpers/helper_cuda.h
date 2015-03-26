/*
 * helper_cuda.h 26-03-2015 Karol Dzitkowski
 */

#ifndef DDJ_HELPER_CUDA_H_
#define DDJ_HELPER_CUDA_H_

#include "../core/logger.h"
#include "../core/config.h"
#include <cuda_runtime_api.h>

namespace ddj {

class HelperCuda {
private:
	/* LOGGER & CONFIG */
	Logger _logger;
	Config* _config;

public:
	HelperCuda()
        : _logger(Logger::getRoot()), _config(Config::GetInstance()) { }

	int    CudaGetDevicesCount();
	bool   CudaCheckDeviceForRequirements(int n);
	int    CudaGetDevicesCountAndPrint();
	void   GetMemoryCount(size_t* freeMemory, size_t* totalMemory);
	void   CudaFreeMemory(void* devPtr);
	int    SetCudaDeviceWithMaxFreeMem();
    cudaError_t CudaAllocateArray(size_t size, void** array);
};

} /* namespace ddj */
#endif /* DDJ_HELPER_CUDA_H_ */
