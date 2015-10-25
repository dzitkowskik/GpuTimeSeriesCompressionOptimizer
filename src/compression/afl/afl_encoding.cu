#include "afl_encoding.hpp"
#include "afl_encoding_impl.cuh"
#include "util/statistics/cuda_array_statistics.hpp"
#include <thrust/device_vector.h>

namespace ddj
{

template<typename T>
int AflEncoding::getMinBitCnt(SharedCudaPtr<T> data)
{
	auto minMax = CudaArrayStatistics().MinMax(data);
	int result = 32;
	if (std::get<0>(minMax) >= 0)
		result = ALT_BITLEN(std::get<1>(minMax));
	return result;
}

template<>
SharedCudaPtr<char> AflEncoding::Encode(SharedCudaPtr<int> data)
{
	int minBit = getMinBitCnt<int>(data);
	int compressed_data_size = (minBit * data->size() + 7) / 8; // in bytes
	auto result = CudaPtr<char>::make_shared(compressed_data_size + sizeof(int));
	run_afl_compress_gpu<int, 32>(
		minBit, data->get(), (int*)(result->get()+sizeof(int)), data->size());
	CUDA_CALL( cudaMemcpy(result->get(), &minBit, sizeof(int), CPY_HTD) );
	cudaDeviceSynchronize();
	return result;
}

template<>
SharedCudaPtr<int> AflEncoding::Decode(SharedCudaPtr<char> data)
{
	thrust::device_ptr<int> data_ptr((int*)data->get());
	int minBit = data_ptr[0];
	int length = (data->size() - sizeof(int)) * 8 / minBit;
	auto result = CudaPtr<int>::make_shared(length);
	run_afl_decompress_gpu<int, 32>(
		minBit, (int*)(data->get()+sizeof(int)), result->get(), length);
	cudaDeviceSynchronize();
	return result;
}

} /* namespace ddj */
