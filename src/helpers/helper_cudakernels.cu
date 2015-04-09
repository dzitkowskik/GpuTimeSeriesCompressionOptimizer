/*
 * helper_cudakernels.cu
 *
 *  Created on: 09-04-2015
 *      Author: Karol Dzitkowski
 */

#include "helper_cudakernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/fill.h>

__global__ void _moduloKernel(int* data, int size, int mod)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	data[idx] %= mod;
}

namespace ddj
{

void HelperCudaKernels::ModuloKernel(int* data, int size, int mod)
{
	const int tpb = 32;
	int blocks = (size + tpb - 1) / tpb;
	_moduloKernel<<<blocks, tpb>>>(data, size, mod);
}

void HelperCudaKernels::ModuloThrust(int* data, int size, int mod)
{
	thrust::device_vector<int> d_mod(size);
	thrust::device_ptr<int> dp(data);
	thrust::fill(d_mod.begin(), d_mod.end(), mod);
	thrust::transform(dp, dp+size, d_mod.begin(), dp, thrust::modulus<int>());
}

} /* namespace ddj */
