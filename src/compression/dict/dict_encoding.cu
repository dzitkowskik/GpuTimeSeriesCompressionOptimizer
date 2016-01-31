#include "dict_encoding.hpp"
#include "core/cuda_macros.cuh"
#include "core/cuda_launcher.cuh"

#include "util/stencil/stencil.hpp"
#include "util/histogram/histogram.hpp"
#include "compression/unique/unique_encoding.hpp"
#include "util/other/cuda_array_reduce.cuh"

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
	CUDA_ASSERT_RETURN( cudaGetLastError() );
    LOG4CPLUS_INFO_FMT(_logger, "DICT encoding START: data size = %lu", data->size());

	if(data->size() <= 0)
		return SharedCudaPtrVector<char>{
			CudaPtr<char>::make_shared(),
			CudaPtr<char>::make_shared(),
			CudaPtr<char>::make_shared()};

	auto mostFrequent = Histogram().GetMostFrequent(data, this->_freqCnt);
    auto mostFrequentStencil = GetMostFrequentStencil(data, mostFrequent);
    auto splittedData = this->_splitter.Split(data, mostFrequentStencil);
    auto packedMostFrequentStencil = Stencil(mostFrequentStencil).pack();
    auto mostFrequentCompressed = UniqueEncoding().CompressUnique(std::get<0>(splittedData), mostFrequent);
    auto otherData = MoveSharedCudaPtr<T, char>(std::get<1>(splittedData));

	LOG4CPLUS_TRACE_FMT(
		_logger,
		"DICT ENCODED output[0] size = %lu, output[1] size = %lu, output[2] size = %lu",
	 	packedMostFrequentStencil->size(), mostFrequentCompressed->size(), otherData->size()
	);

	CUDA_ASSERT_RETURN( cudaGetLastError() );
	LOG4CPLUS_INFO(_logger, "DICT enoding END");

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
	LOG4CPLUS_INFO_FMT(
		_logger,
		"DICT decoding START: input[0] size = %lu, input[1] size = %lu, input[2] size = %lu",
		input[0]->size(), input[1]->size(), input[2]->size()
	);

	if(input[1]->size() <= 0 && input[2]->size() <= 0)
		return CudaPtr<T>::make_shared();

	// UNPACK STENCIL
	auto stencil = Stencil(input[0]);
	auto mostFrequentCompressed = input[1];
	auto other = MoveSharedCudaPtr<char, T>(input[2]);

	// DECOMPRESS MOST FREQUENT
	auto mostFrequent = UniqueEncoding().DecompressUnique<T>(mostFrequentCompressed);

	// MERGE DATA
	auto result = this->_splitter.Merge<T>(std::make_tuple(mostFrequent, other), *stencil);

	CUDA_ASSERT_RETURN( cudaGetLastError() );
    LOG4CPLUS_INFO(_logger, "DICT decoding END");

	return result;
}

size_t DictEncoding::GetCompressedSize(SharedCudaPtr<char> data, DataType type)
{
	if(data->size() <= 0) return 0;
	switch(type)
	{
		case DataType::d_int:
			return GetCompressedSize(boost::reinterpret_pointer_cast<CudaPtr<int>>(data));
		case DataType::d_float:
			return GetCompressedSize(boost::reinterpret_pointer_cast<CudaPtr<float>>(data));
		default:
			throw NotImplementedException("No DictEncoding::GetCompressedSize implementation for that type");
	}
}

template<typename T>
size_t DictEncoding::GetCompressedSize(SharedCudaPtr<T> data)
{
	if(data->size() <= 0) return 0;
	auto mostFrequent = Histogram().GetMostFrequent(data, this->_freqCnt);
	int freqCnt = mostFrequent->size();
	auto mostFrequentStencil = GetMostFrequentStencil(data, mostFrequent);
	size_t mostFrequentCompressedSize = 2*sizeof(size_t) + freqCnt*sizeof(T);
	int dataPerOutputCnt = (8 * sizeof(unsigned int)) / ALT_BITLEN(freqCnt - 1);
	int stencilDataCnt = reduce_thrust(mostFrequentStencil, thrust::plus<int>());
	int outputSize = (stencilDataCnt + dataPerOutputCnt - 1) / dataPerOutputCnt;
	mostFrequentCompressedSize += outputSize * sizeof(unsigned int);
	int othersCnt = data->size() - stencilDataCnt;
	size_t otherDataSize = othersCnt * sizeof(T);
	return mostFrequentCompressedSize + otherDataSize;
}

#define DICT_ENCODING_SPEC(X) \
	template SharedCudaPtrVector<char> DictEncoding::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> DictEncoding::Decode<X>(SharedCudaPtrVector<char> data);
FOR_EACH(DICT_ENCODING_SPEC, short, char, double, float, int, long, long long, unsigned int)

} /* namespace ddj */
