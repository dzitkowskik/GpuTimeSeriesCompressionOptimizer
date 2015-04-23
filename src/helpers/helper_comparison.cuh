/*
 * helper_comparison.cuh 26-03-2015 Karol Dzitkowski
 */

#ifndef DDJ_HELPER_COMPARISON_H_
#define DDJ_HELPER_COMPARISON_H_

#include "helper_macros.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define COMP_THREADS_PER_BLOCK 512

template <typename T>
__global__ void _compareElementsKernel(T* a, T*b, int size, bool* out)
{
	unsigned int iElement = blockDim.x * blockIdx.x + threadIdx.x;
	if (iElement >= size) return;
	out[iElement] = a[iElement] != b[iElement];
}

template <typename T>
__host__ bool CompareDeviceArrays(T* a, T* b, int size)
{
    bool* out;
    CUDA_CALL(cudaMalloc((void**)&out, size*sizeof(T)));
    int blocks = (size + COMP_THREADS_PER_BLOCK - 1) / COMP_THREADS_PER_BLOCK;
    _compareElementsKernel<<<blocks, COMP_THREADS_PER_BLOCK>>>(a, b, size, out);
    cudaDeviceSynchronize();
    thrust::device_vector<bool> out_vector(out, out+size);
    thrust::inclusive_scan(out_vector.begin(), out_vector.end(), out_vector.begin());
    int result = out_vector.back();
    CUDA_CALL(cudaFree(out));
    return result == 0;
}

__host__ bool CompareDeviceFloatArrays(float* a, float* b, int size);

#endif
