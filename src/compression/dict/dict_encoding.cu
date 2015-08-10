#include "dict_encoding.hpp"
#include "core/cuda_macros.cuh"
#include "util/histogram/cuda_histogram.hpp"
#include "helpers/helper_cuda.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

namespace ddj {

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
        if(value == mostFrequent[i])
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

__global__ void CompressMostFrequentKernel(
    int* data,
    int dataSize,
    int bitsNeeded,
    int dataPerOutputCnt,
    int* mostFrequent,
    int freqCnt,
    unsigned int* output,
    int outputSize)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // output index
    if(idx >= outputSize) return;

    unsigned int result = 0;
    int value = 0;

    for(int i = 0; i < dataPerOutputCnt; i++)
    {
        value = data[idx * dataPerOutputCnt + i];
        for(int j = 0; j < freqCnt; j++)
        {
            if(value == mostFrequent[j])
            {
                result = SaveNbitIntValToWord(bitsNeeded, i, j, result);
            }
        }
    }
    output[idx] = result;
}

SharedCudaPtr<char> DictEncoding::CompressMostFrequent(
    SharedCudaPtr<int> data, SharedCudaPtr<int> mostFrequent)
{
    int cnt = mostFrequent->size();
    int bitsNeeded = ALT_BITLEN(cnt-1);
    int outputItemBitSize = 8 * sizeof(unsigned int);
    int dataPerOutputCnt = outputItemBitSize / bitsNeeded;
    int outputSize = (data->size() + dataPerOutputCnt - 1) / dataPerOutputCnt;
    int outputSizeInBytes = outputSize * sizeof(unsigned int);
    int mostFrequentSizeInBytes = mostFrequent->size() * sizeof(int);
    auto result = CudaPtr<char>::make_shared(outputSizeInBytes + mostFrequentSizeInBytes);

    this->_policy.setSize(outputSize);
    cudaLaunch(this->_policy, CompressMostFrequentKernel,
        data->get(),
        data->size(),
        bitsNeeded,
        dataPerOutputCnt,
        mostFrequent->get(),
        mostFrequent->size(),
        (unsigned int*)(result->get()+mostFrequentSizeInBytes),
        outputSize);

    CUDA_CALL( cudaMemcpy(result->get(), mostFrequent->get(), mostFrequentSizeInBytes, CPY_DTD) );
    cudaDeviceSynchronize();
    return result;
}




//     1. CREATE HISTOGRAM
//
//     2. GET N MOST FREQUENT VALUES
//
//     3. SPLIT TO MOST FREQUENT AND OTHERS
//
//     4. PACK STENCIL
//
//     5. COMPRESS MOST FREQUENT
//          a) GET DISTINCT NUMBERS AND GIVE THEM THE SHORTEST UNIQUE KEYS POSSIBLE
//          b) LEAVE N UNIQUE VALUES AT BEGINNING
//          c) REPLACE OTHER OCCURENCES OF THESE NUMBERS BY THEIR CODES (GREY CODE)

} /* namespace ddj */
