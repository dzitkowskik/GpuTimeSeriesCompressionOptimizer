/*
 *  simple_patch.cu
 *
 *  Created on: 13-05-2015
 *      Author: Karol Dzitkowski
 */

#include "patch.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include "helpers/helper_macros.h"

#define SPLIT_ENCODING_GPU_BLOCK_SIZE 64

namespace ddj {

using namespace std;

template<typename T, typename UnaryOperator>
__global__ void stencilKernel(T* data, int size, int* out, UnaryOperator op)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= size) return;
    T value = data[idx];
    out[idx] = op(value) ? 1 : 0;
}

template<typename DT, typename OPT>
void PatchedData<DT, OPT>::Init(SharedCudaPtr<DT> data)
{
    int size = data->size();
	int block_size = SPLIT_ENCODING_GPU_BLOCK_SIZE;
	int block_cnt = (size + block_size - 1) / block_size;

    this->_stencil = CudaPtr<int>::make_shared(size);
    stencilKernel<DT, OPT><<<block_size, block_cnt>>>(
        data->get(), data->size(), this->_stencil->get(), _op);

    return SplitKernel(data, this->_stencil);
}

} /* namespace ddj */
