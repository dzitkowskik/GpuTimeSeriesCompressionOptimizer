#include "dict_encoding.hpp"
#include "core/cuda_macros.cuh"
#include "util/histogram/cuda_histogram.hpp"
#include "helpers/helper_cuda.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include "util/histogram/cuda_histogram.hpp"

namespace ddj {

SharedCudaPtr<int> DictEncoding::GetMostFrequent(SharedCudaPtrPair<int, int> histogram, int freqCnt)
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


// MOST FREQUENT COMPRESSION ALGORITHM
//
// As the first block we save an array of most frequent values
// Then we compress all data (values from a set of most frequent values) using N bits
// N is the smallest number of bits we can use to encode most frequent array length
// We encode this way each value as it's index in most frequent values array. We save
// as many values as possible in unsigned int values. For example if we need 4 bits to
// encode single value, we put 8 values in one unsigned int and store it in output table.
// We call that single unsigned int value an unit.
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
    int cnt = mostFrequent->size();                         // how many distinct items to encode
    int bitsNeeded = ALT_BITLEN(cnt-1);                     // min bits needed to encode
    int outputItemBitSize = 8 * sizeof(unsigned int);       // how many bits are in output unit
    int dataPerOutputCnt = outputItemBitSize / bitsNeeded;  // how many items will be encoded in single unit
    int outputSize = (data->size() + dataPerOutputCnt - 1) / dataPerOutputCnt;  // output units cnt
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


// MOST FREQUENT DECOMPRESSION ALGORITHM
//
// Having the number of most frequent values we restore the original array of most freq values
// Then we count how many unsigned int values fit in encoded data and what is the minimal bit
// number to encode the number of most frequent values. Using that we can compute how many values
// are stored in sigle unsigned int number. Then we take in parallel each unsigned int and decode it
// as the values from the array at index equal to that N bit block stored in unsigned int number
// casted to integer. We call that single unsigned int value an unit.
__global__ void DecompressMostFrequentKernel(
    unsigned int* data,
    int* mostFrequent,
    const int size,
    int* output,
    const int bitsNeeded,
    const int dataPerUnitCnt
    )
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // output index
    if(idx >= size) return;
    int value, index;
    for(int i=0; i<dataPerUnitCnt; i++)
    {
        index = ReadNbitIntValFromWord(bitsNeeded, i, data[idx]);
        value = mostFrequent[index];
        output[idx * dataPerUnitCnt + i] = value;
    }
}

SharedCudaPtr<int> DictEncoding::DecompressMostFrequent(SharedCudaPtr<char> data, int freqCnt)
{
    int bitsNeeded = ALT_BITLEN(freqCnt-1);                 // min bit cnt to encode freqCnt values
    int outputItemBitSize = 8 * sizeof(unsigned int);       // single encoded unit size
    int dataPerUnitCnt = outputItemBitSize / bitsNeeded;    // how many items are in one unit
    int unitCnt =  data->size() / outputItemBitSize;        // how many units are in data
    int outputSize = unitCnt * dataPerUnitCnt;              // how many items were compressed
    int mostFrequentSizeInBytes = freqCnt * sizeof(int);    // size in bytes of most frequent array
    auto result = CudaPtr<int>::make_shared(outputSize);
    this->_policy.setSize(unitCnt);
    cudaLaunch(this->_policy, DecompressMostFrequentKernel,
        (unsigned int*)(data->get()+mostFrequentSizeInBytes),
        (int*)data->get(),
        unitCnt,
        result->get(),
        bitsNeeded,
        dataPerUnitCnt);

    cudaDeviceSynchronize();
    return result;
}
 
// DICT ENCODING ALGORITHM
//
//  1. CREATE HISTOGRAM
//  2. GET N MOST FREQUENT VALUES
//  3. SPLIT TO MOST FREQUENT AND OTHERS
//  4. PACK STENCIL
//  5. COMPRESS MOST FREQUENT
//       a) GET DISTINCT NUMBERS AND GIVE THEM THE SHORTEST UNIQUE KEYS POSSIBLE
//       b) LEAVE N UNIQUE VALUES AT BEGINNING
//       c) REPLACE OTHER OCCURENCES OF THESE NUMBERS BY THEIR CODES (GREY CODE)
//  6. RETURN A PAIR OF ARRAYS (MOST FREQUENT (COMPRESSED), OTHERS (UNCOMPRESSED))

//SharedCudaPtrVector<char> DictEncoding::Encode(SharedCudaPtr<int> data)
//{
//    auto histogram = CudaHistogram.IntegerHistogram(data);
//    auto mostFrequent = GetMostFrequent(histogram, 4);
//    auto mostFrequentStencil = GetMostFrequentStencil(data, mostFrequent);
//    auto splittedData = _cudaKernels.SplitKernel(data, mostFrequentStencil);
//    auto packedMostFrequentStencil = Stencil(std::get<0>(splittedData)).pack();
//    return NULL;
//}



} /* namespace ddj */
