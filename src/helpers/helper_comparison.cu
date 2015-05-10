#include "helper_comparison.cuh"
#include "compression/macros.cuh"
#include "helper_macros.h"
#include <boost/type_traits/is_same.hpp>

#define MAX_FLOAT_DIFF 0.000001f

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
	out[iElement] =
			!_floatsEqual(a[iElement], b[iElement], 5) &&
			abs(a[iElement] - b[iElement]) > MAX_FLOAT_DIFF;
}

template <typename T>
__global__ void _compareElementsKernel(T* a, T*b, int size, bool* out)
{
	unsigned int iElement = blockDim.x * blockIdx.x + threadIdx.x;
	if (iElement >= size) return;
	out[iElement] = a[iElement] != b[iElement];
}

template <typename T> bool CompareDeviceArrays(T* a, T* b, int size)
{
    bool* out;
    CUDA_CALL(cudaMalloc((void**)&out, size*sizeof(T)));
    int blocks = (size + COMP_THREADS_PER_BLOCK - 1) / COMP_THREADS_PER_BLOCK;
    if(boost::is_same<T, float>::value)
    	_compareFloatsKernel<<<blocks, COMP_THREADS_PER_BLOCK>>>((float*)a, (float*)b, size, out);
    else
    	_compareElementsKernel<<<blocks, COMP_THREADS_PER_BLOCK>>>(a, b, size, out);
    cudaDeviceSynchronize();
    thrust::device_vector<bool> out_vector(out, out+size);
    thrust::inclusive_scan(out_vector.begin(), out_vector.end(), out_vector.begin());
    int result = out_vector.back();
    CUDA_CALL(cudaFree(out));
    return result == 0;
}

#define COMP_SPEC(X) template bool CompareDeviceArrays <X> (X* a, X* b, int size);
FOR_EACH(COMP_SPEC, double, float, int, long, long long, unsigned int, unsigned long, unsigned long long)

