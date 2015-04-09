#include "afl.cuh"
#include "afl_gpu.cuh"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>

namespace ddj
{

int AFLCompression::getMinBitCnt(int* data, int size)
{
	thrust::device_ptr<int> dp(data);
	auto tuple = thrust::minmax_element(dp, dp+size);
	int min = *(tuple.first);
	int max = *(tuple.second);

	if (min < 0) return 32;
	else return ALT_BITLEN(max);
}

void* AFLCompression::Encode(int* data, int in_size, int& out_size, AFLCompressionMetadata& metadata)
{
	int min_bit = getMinBitCnt(data, in_size); // in bits
	int* dev_out;
	int compressed_data_size = ((min_bit * in_size) / 8) + 1; // in bytes
	cudaMalloc((void **)&dev_out, compressed_data_size);
	run_afl_compress_gpu<int, 32>(min_bit, data, dev_out, in_size);
	out_size = compressed_data_size;
	metadata.min_bit = min_bit;
	return dev_out;
}

void* AFLCompression::Decode(void* data, int in_size, int& out_size, AFLCompressionMetadata metadata)
{
	int* dev_out;
	int min_bit = metadata.min_bit;
	int length = (in_size * 8) / min_bit;
	int decompressed_data_size = length * sizeof(int);
	cudaMalloc((void **)&dev_out, decompressed_data_size);
	run_afl_compress_gpu<int, 32>(min_bit, (int*)data, dev_out, length);
	out_size = length;
	return dev_out;
}

} /* namespace ddj */
