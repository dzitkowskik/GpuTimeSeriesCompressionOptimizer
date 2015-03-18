/*
 * CudaCommon.h
 *
 *  Created on: 19-11-2013
 *      Author: ghash
 */

#ifndef CUDACOMMON_H_
#define CUDACOMMON_H_

#include "logger.h"
#include "config.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CUDA_CHECK_RETURN(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        fprintf(stderr, "Error %s at line %d in file %s\n",                 \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);       \
        exit(1);                                                            \
    } }

namespace ddj {

class CudaCommons {
private:
	/* LOGGER & CONFIG */
	Logger _logger;
	Config* _config;

public:
	CudaCommons()
        : _logger(Logger::getRoot()), _config(Config::GetInstance()) { }

	/* CUDA DEVICES */
	int CudaGetDevicesCount();
	bool CudaCheckDeviceForRequirements(int n);
	int CudaGetDevicesCountAndPrint();
	void GetMemoryCount(size_t* freeMemory, size_t* totalMemory);

	/* CUDA MALLOC */
	cudaError_t CudaAllocateArray(size_t size, void** array);
	void CudaFreeMemory(void* devPtr);
	int SetCudaDeviceWithMaxFreeMem();
};

} /* namespace ddj */
#endif /* CUDACOMMON_H_ */
