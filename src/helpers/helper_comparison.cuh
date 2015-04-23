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

// ulp = units in the last place; maxulps = maximum number of
// representable floating point numbers by which x and y may differ.
__host__ __device__ bool _floatsEqual(float a, float b, int maxulps)
{
	// convert to integer.
	int aint = *(int*)&a;
	int bint = *(int*)&b;
	// make lexicographically ordered as a twos-complement int
	if (aint < 0) aint = 0x80000000 - aint;
	if (bint < 0) bint = 0x80000000 - bint;
	// compare.
	return abs(aint - bint) <= maxulps;
}

__global__ void _compareFloatsKernel(float* a, float* b, int size, bool* out)
{
	unsigned int iElement = blockDim.x * blockIdx.x + threadIdx.x;
	if (iElement >= size) return;
	out[iElement] = !_floatsEqual(a[iElement], b[iElement], 5);
}

__host__ bool CompareDeviceFloatArrays(float* a, float* b, int size)
{
    bool* out;
    CUDA_CALL(cudaMalloc((void**)&out, size*sizeof(float)));
    int blocks = (size + COMP_THREADS_PER_BLOCK - 1) / COMP_THREADS_PER_BLOCK;
    _compareFloatsKernel<<<blocks, COMP_THREADS_PER_BLOCK>>>(a, b, size, out);
    cudaDeviceSynchronize();
    thrust::device_vector<bool> out_vector(out, out+size);
    thrust::inclusive_scan(out_vector.begin(), out_vector.end(), out_vector.begin());
    int result = out_vector.back();
    CUDA_CALL(cudaFree(out));
    return result == 0;
}

#endif
