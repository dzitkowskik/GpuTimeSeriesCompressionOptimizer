#include "afl.cuh"
#include "afl_gpu.cuh"

namespace ddj
{

void* AFLCompression::Encode(int* data, int in_size, int& out_size)
{
	int* dev_out;
	int cword = sizeof(int) * 8;
	int compressed_data_size = (max_size < cword  ? cword : max_size) * sizeof(int);
	unsigned long data_size = max_size * sizeof(int);
	cudaMalloc((void **)&dev_out, compressed_data_size);
	run_afl_compress_gpu<int, 32>(cword, data, dev_out, max_size);
	return dev_out;
}

void* AFLCompression::Decode(int* data, int in_size, int& out_size)
{
	int* dev_out;
	int cword = sizeof(int) * 8;
	int compressed_data_size = (max_size < cword  ? cword : max_size) * sizeof(int);
	unsigned long data_size = max_size * sizeof(int);
	cudaMalloc((void **)&dev_out, compressed_data_size);
	run_afl_compress_gpu<int, 32>(cword, data, dev_out, max_size);
	return dev_out;
}

} /* namespace ddj */
