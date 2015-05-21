/*
 *  simple_patch.cu
 *
 *  Created on: 13-05-2015
 *      Author: Karol Dzitkowski
 */

#include "simple_patch.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include "helpers/helper_macros.h"

#define SPLIT_ENCODING_GPU_BLOCK_SIZE 64

namespace ddj {

using namespace std;

template<typename T>
__global__ void splitPrepareKernel(T* data, int size, int* out, T low, T high)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= size) return;
    T value = data[idx];
    if(value > high || value < low) out[idx] = 1;
    else out[idx] = 0;
}

template<typename T>
tuple<SharedCudaPtr<T>, SharedCudaPtr<T>> SimplePatch<T>::split(SharedCudaPtr<T> data)
{
    int size = data->size();
	int block_size = SPLIT_ENCODING_GPU_BLOCK_SIZE;
	int block_cnt = (size + block_size - 1) / block_size;

    auto stencil = CudaPtr<int>::make_shared(size);
    splitPrepareKernel<T><<<block_size, block_cnt>>>(
        data->get(), data->size(), stencil->get(), _low, _high);

    thrust::device_ptr<T> data_ptr(data->get());
    thrust::device_ptr<int> stencil_ptr(stencil->get());
    thrust::stable_sort_by_key(stencil_ptr, stencil_ptr+size, data_ptr);

    int b_size = thrust::reduce(stencil_ptr, stencil_ptr + size, 0);
    int a_size = size - b_size;

    auto a = CudaPtr<T>::make_shared(a_size);
    auto b = CudaPtr<T>::make_shared(b_size);
    a->fill(data->get(), a_size);
    b->fill(data->get()+a_size, b_size);

    return std::make_tuple(a, b);
}

template<typename T>
SharedCudaPtr<char> SimplePatch<T>::partition(SharedCudaPtr<T> data)
{
    int size = data->size();
    size_t byte_size = data->size() * sizeof(T);
	int block_size = SPLIT_ENCODING_GPU_BLOCK_SIZE;
	int block_cnt = (size + block_size - 1) / block_size;

    auto stencil = CudaPtr<T>::make_shared(size);
    splitPrepareKernel<T><<<block_size, block_cnt>>>(
        data->get(), data->size(), stencil->get(), _low, _high);
    thrust::device_ptr<T> data_ptr(data->get());
    thrust::device_ptr<int> stencil_ptr(stencil->get());
    auto result = CudaPtr<char>::make_shared(byte_size + sizeof(int));
    cudaDeviceSynchronize();

    thrust::stable_sort_by_key(stencil_ptr, stencil_ptr+size, data_ptr);
    int b_size = thrust::reduce(stencil_ptr, stencil_ptr + size, 0);
    int a_size = size - b_size;

    auto res_data = result->get()+sizeof(int);
    CUDA_CALL( cudaMemcpy(res_data, data->get(), byte_size, CPY_DTD) );
    CUDA_CALL( cudaMemcpy(result->get(), &a_size, sizeof(int), CPY_HTD) );

    return result;
}

} /* namespace ddj */
