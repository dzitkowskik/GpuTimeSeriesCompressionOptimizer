/*
 *  simple_patch.cu
 *
 *  Created on: 13-05-2015
 *      Author: Karol Dzitkowski
 */

#include "patch.cuh"
#include "helpers/helper_cuda.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include "helpers/helper_macros.h"
#include "core/operators.cuh"

#define SPLIT_ENCODING_GPU_BLOCK_SIZE 64

namespace ddj {

using namespace std;

template<typename DataType, typename UnaryOperator>
PatchedData<DataType, UnaryOperator>::PatchedData(UnaryOperator op)
{
    this->_op = op;
}

template<typename DataType, typename UnaryOperator>
PatchedData<DataType, UnaryOperator>::~PatchedData()
{
}

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

    // Create stencil
    this->_stencil = CudaPtr<int>::make_shared(size);
    this->_policy.setSize(size);
    cudaLaunch(this->_policy, stencilKernel<DT, OPT>,
        data->get(), data->size(), this->_stencil->get(), _op);

    // Split according to the stencil
    this->_data = this->_kernels.SplitKernel(data, this->_stencil);
}

template<typename DT, typename OPT>
SharedCudaPtr<DT> PatchedData<DT, OPT>::GetFirst()
{
    return get<0>(this->_data);
}

template<typename DT, typename OPT>
SharedCudaPtr<DT> PatchedData<DT, OPT>::GetSecond()
{
    return get<1>(this->_data);
}

template class PatchedData<int, OutsideOperator<int>>;

} /* namespace ddj */
