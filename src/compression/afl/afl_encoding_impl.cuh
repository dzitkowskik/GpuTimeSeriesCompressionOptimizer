#ifndef DDJ_AFL_ENCODING_IMPL_CUH_
#define DDJ_AFL_ENCODING_IMPL_CUH_

#include "core/cuda_macros.cuh"
#include "core/macros.h"
#include <stdio.h>

// CWARP_SIZE is given in template to allow the compiler to optimize it (i.e. possible values are 32 or 1)
// CWARP_SIZE possible values:
//  * 32 aligned access - fast
//  * unaligned access - slower

// Currently supported types: int, unsigned int, long, unsigned long

#define FL_ALGORITHM_MOD_FL 1
#define FL_ALGORITHM_MOD_AFL 32
#define CWORD_SIZE(T)(T) sizeof(T) * 8

template <typename T, char CWARP_SIZE>
__host__ void run_afl_decompress_gpu(int bit_length, T *data, T *compressed_data, unsigned long length);

template <typename T, char CWARP_SIZE>
__host__ void run_afl_compress_gpu(int bit_length, T *compressed_data, T *decompressed_data, unsigned long length);

template <typename T, char CWARP_SIZE>
__global__ void afl_compress_gpu(int bit_length, T *data, T *compressed_data, unsigned long length);

template <typename T, char CWARP_SIZE>
__global__ void afl_decompress_gpu(int bit_length, T *compressed_data, T * decompress_data, unsigned long length);

template <typename T, char CWARP_SIZE>
__device__ __host__ void afl_compress_base_gpu(
		int bit_length,
		unsigned long data_id,
		unsigned long comp_data_id,
		T *data,
		T *compressed_data,
		unsigned long length);

template <typename T, char CWARP_SIZE>
__device__ __host__ void afl_decompress_base_gpu(
		int bit_length,
		unsigned long comp_data_id,
		unsigned long data_id,
		T *compressed_data,
		T *data,
		unsigned long length);


template < typename T, char CWARP_SIZE >
__host__ void run_afl_compress_gpu(int bit_length, T *data, T *compressed_data, unsigned long length)
{
    int block_size = CWARP_SIZE * 8; // better occupancy
    unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    afl_compress_gpu <T, CWARP_SIZE> <<<block_number, block_size>>> (bit_length, data, compressed_data, length);
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_decompress_gpu(int bit_length, T *compressed_data, T *data, unsigned long length)
{
    int block_size = CWARP_SIZE * 8; // better occupancy
    unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    afl_decompress_gpu <T, CWARP_SIZE> <<<block_number, block_size>>> (bit_length, compressed_data, data, length);
}

template < typename T, char CWARP_SIZE >
__global__ void afl_compress_gpu (int bit_length, T *data, T *compressed_data, unsigned long length)
{
    unsigned int warp_lane = (threadIdx.x % CWARP_SIZE);
    unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
    unsigned long cdata_id = data_block * bit_length + warp_lane;

    afl_compress_base_gpu <T, CWARP_SIZE> (bit_length, data_id, cdata_id, data, compressed_data, length);
}

template < typename T, char CWARP_SIZE >
__global__ void afl_decompress_gpu (int bit_length, T *compressed_data, T * decompress_data, unsigned long length)
{
    unsigned int warp_lane = (threadIdx.x % CWARP_SIZE);
    unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
    unsigned long cdata_id = data_block * bit_length + warp_lane;

    afl_decompress_base_gpu <T, CWARP_SIZE> (bit_length, cdata_id, data_id, compressed_data, decompress_data, length);
}

template <typename T, char CWARP_SIZE>
__device__  __host__ void afl_compress_base_gpu(
		int bit_length,
		unsigned long data_id,
		unsigned long comp_data_id,
		T *data,
		T *compressed_data,
		unsigned long length)
{
    T v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;

    for (unsigned int i = 0; (i < CWORD_SIZE(T)) && (pos_data < length); ++i)
    {
        v1 = data[pos_data];
        pos_data += CWARP_SIZE;

        if (v1_pos >= CWORD_SIZE(T) - bit_length)
        {
            v1_len = CWORD_SIZE(T) - v1_pos;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);

            compressed_data[pos] = value;

            v1_pos = bit_length - v1_len;
            value = GETNPBITS(v1, v1_pos, v1_len);

            pos += CWARP_SIZE;
        }
        else
        {
            v1_len = bit_length;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);
            v1_pos += v1_len;
        }
    }
    if (pos_data >= length  && pos_data < length + CWARP_SIZE && pos < length)
    {
        compressed_data[pos] = value;
    }
}

template <typename T, char CWARP_SIZE>
__device__ __host__ void afl_decompress_base_gpu(
		int bit_length,
		unsigned long comp_data_id,
		unsigned long data_id,
		T *compressed_data,
		T *data,
		unsigned long length)
{
    unsigned long pos = comp_data_id, pos_decomp = data_id;
    unsigned int v1_pos = 0, v1_len;
    T v1, ret;

    if (pos_decomp > length || pos > length) // Decompress not more elements then length
        return;
    v1 = compressed_data[pos];
    for (unsigned int i = 0; (i < CWORD_SIZE(T)) && (pos_decomp < length); ++i)
    {
        if (v1_pos >= CWORD_SIZE(T) - bit_length)
        {
            v1_len = CWORD_SIZE(T) - v1_pos;
            ret = GETNPBITS(v1, v1_len, v1_pos);

            pos += CWARP_SIZE;
            if(pos < length)
			{
				v1 = compressed_data[pos];
				v1_pos = bit_length - v1_len;
				ret = ret | ((GETNBITS(v1, v1_pos))<< v1_len);
            }
        }
        else
        {
            v1_len = bit_length;
            ret = GETNPBITS(v1, v1_len, v1_pos);
            v1_pos += v1_len;
        }

        data[pos_decomp] = ret;
        pos_decomp += CWARP_SIZE;
    }
}

#endif /* DDJ_AFL_ENCODING_IMPL_CUH_ */
