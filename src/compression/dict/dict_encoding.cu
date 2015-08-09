#include "dict_encoding.hpp"
#include "util/histogram/cuda_histogram.hpp"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

SharedCudaPtr<int> GetMostFrequent(SharedCudaPtrPair<int, int> histogram, int freqCnt)
{
    thrust::device_ptr<int> keys_ptr(histogram.first->get());
    thrust::device_ptr<int> counts_ptr(histogram.second->get());
    int N = histogram.first->size();

    // sort to have greater counts first
    thrust::sort_by_key(counts_ptr, counts_ptr + N, keys_ptr, thrust::greater<int>());

    // get first freqCnt keys
    auto result = CudaPtr<int>::make_shared(freqCnt);
    thrust::device_ptr<int> result_ptr(result->get());
    thrust::copy_n(keys_ptr, freqCnt, result_ptr);

    return result;
}

__global__ void getMostFrequentStencilKernel(
    int* data, int size, int* mostFrequent, int freqCnt, int* output)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= size) return;
    int value = data[idx];
    output[idx] = 0;
    for(int i = 0; i < freqCnt; i++)
    {
        if(value == modeFrequent[i])
        {
            output[idx] = 1;
        }
    }
}

SharedCudaPtr<int> DictEncoding::GetMostFrequentStencil(
    SharedCudaPtr<int> data, SharedCudaPtr<int> mostFrequent)
{
    auto result = CudaPtr<int>::make_shared(data->size());
    this->_policy.setSize(data->size());
    cudaLaunch(this->_policy, getMostFrequentStencilKernel,
        data->get(), data->size(), mostFrequent->get(), mostFrequent->size(), result->get());

    cudaDeviceSynchronize();
    return result;
}

// SharedCudaPtr<char> Encode(SharedCudaPtr<int> data)
// {
//     // 1. CREATE HISTOGRAM
//     CudaHistogram cudaHistogram;
//     auto histogram = cudaHistogram.IntegerHistogram(data);
//
//     // 2. GET N MOST FREQUENT VALUES
//
//     // 3. SPLIT TO MOST FREQUENT AND OTHERS
//
//     // 4. PACK STENCIL
//
//     // 5. COMPRESS MOST FREQUENT
//          a) GET DISTINCT NUMBERS AND GIVE THEM THE SHORTEST UNIQUE KEYS POSSIBLE
//          b) LEAVE N UNIQUE VALUES AT BEGINNING
//          c) REPLACE OTHER OCCURENCES OF THESE NUMBERS BY THEIR CODES (GREY CODE)
// }
