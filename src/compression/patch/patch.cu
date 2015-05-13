/*
 *  patch.cu
 *
 *  Created on: 13-05-2015
 *      Author: Karol Dzitkowski
 */

#include "patch.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define SPLIT_ENCODING_GPU_BLOCK_SIZE 64

namespace ddj {

    // template<typename T, int N>
    // boost::array<SharedCudaPtr<T>, N> SimplePatch::Split(SharedCudaPtr<T> data)
    // {
    //
    // }
    //
	// template<typename T>
    // SharedCudaPtr<T> SimplePatch::Merge(SharedCudaPtr<char> data)
    // {
    //
    // }

    __global__ void splitPrepareKernel(
        float* data,
        int size,
        int* out,
        float low,
        float high)
    {
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    	if(idx >= size) return;
        float value = data[idx];
        if(value > high || value < low) out[idx] = 1;
        else out[idx] = 0;
    }

    std::tuple<SharedCudaPtr<float>, SharedCudaPtr<float>> SimplePatch::split(
        SharedCudaPtr<float> data, float low, float high)
    {
        int size = data->size();
    	int block_size = SPLIT_ENCODING_GPU_BLOCK_SIZE;
    	int block_cnt = (size + block_size - 1) / block_size;

        auto stencil = CudaPtr<int>::make_shared(size);
        splitPrepareKernel<<<block_size, block_cnt>>>(
            data->get(), data->size(), stencil->get(), low, high);

        thrust::device_ptr<float> data_ptr(data->get());
        thrust::device_ptr<int> stencil_ptr(stencil->get());
        thrust::stable_sort_by_key(stencil_ptr, stencil_ptr+size, data_ptr);

        int b_size = thrust::reduce(stencil_ptr, stencil_ptr + size, 0);
        int a_size = size - b_size;

        auto a = CudaPtr<float>::make_shared(a_size);
        auto b = CudaPtr<float>::make_shared(b_size);
        a->fill(data->get(), a_size);
        b->fill(data->get()+a_size, b_size);

        return std::make_tuple(a, b);
    }

} /* namespace ddj */
