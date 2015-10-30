#include "dict_encoding.hpp"
#include "core/cuda_macros.cuh"
#include "helpers/helper_cuda.cuh"
#include "util/stencil/stencil.hpp"
#include "util/histogram/histogram.hpp"
#include "compression/unique/unique_encoding.hpp"

#include <thrust/device_ptr.h>
#include <thrust/count.h>

namespace ddj {

template<typename T>
__global__ void getMostFrequentStencilKernel(
		T* data,
		int size,
		T* mostFrequent,
		int freqCnt,
		int* output)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= size) return;
    T value = data[idx];
    output[idx] = 0;
    for(int i = 0; i < freqCnt; i++)
    {
        if(value == mostFrequent[i])
        {
            output[idx] = 1;
        }
    }
}

template<typename T>
SharedCudaPtr<int> DictEncoding::GetMostFrequentStencil(
		SharedCudaPtr<T> data,
		SharedCudaPtr<T> mostFrequent)
{
    auto result = CudaPtr<int>::make_shared(data->size());

    this->_policy.setSize(data->size());
    cudaLaunch(this->_policy, getMostFrequentStencilKernel<T>,
        data->get(),
        data->size(),
        mostFrequent->get(),
        mostFrequent->size(),
        result->get());

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
//  6. RETURN A VECTOR (STENCIL, MOST FREQUENT (COMPRESSED), OTHERS (UNCOMPRESSED))
template<typename T>
SharedCudaPtrVector<char> DictEncoding::Encode(SharedCudaPtr<T> data)
{
	auto mostFrequent = Histogram().GetMostFrequent(data, this->_freqCnt);
    auto mostFrequentStencil = GetMostFrequentStencil(data, mostFrequent);
    auto splittedData = this->_splitter.Split(data, mostFrequentStencil);
    auto packedMostFrequentStencil = Stencil(mostFrequentStencil).pack();
    auto mostFrequentCompressed = UniqueEncoding().CompressUnique(std::get<0>(splittedData), mostFrequent);
    auto otherData = MoveSharedCudaPtr<T, char>(std::get<1>(splittedData));
    return SharedCudaPtrVector<char> {packedMostFrequentStencil, mostFrequentCompressed, otherData};
}

// DICT DECODING ALGORITHM
//
// 1. UNPACK STENCIL
// 2. GET MOST FREQUENT DATA COMPRESSED AND DECOMPRESS IT
// 3. USE STENCIL TO MERGE MOST FREQUENT DATA AND OTHER
// 4. RETURN MERGED DATA
template<typename T>
SharedCudaPtr<T> DictEncoding::Decode(SharedCudaPtrVector<char> input)
{
	// UNPACK STENCIL
	auto stencil = Stencil(input[0]);
	auto mostFrequentCompressed = input[1];
	auto other = MoveSharedCudaPtr<char, T>(input[2]);

	// DECOMPRESS MOST FREQUENT
	auto mostFrequent = UniqueEncoding().DecompressUnique<T>(mostFrequentCompressed);

	// MERGE DATA
	return this->_splitter.Merge<T>(std::make_tuple(mostFrequent, other), *stencil);
}

#define DICT_ENCODING_SPEC(X) \
	template SharedCudaPtrVector<char> DictEncoding::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> DictEncoding::Decode<X>(SharedCudaPtrVector<char> data);
FOR_EACH(DICT_ENCODING_SPEC, float, int, long, long long, unsigned int)

} /* namespace ddj */
