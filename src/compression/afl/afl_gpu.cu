#include "afl_gpu.cuh"
#include "core/cuda_macros.cuh"
#include "core/macros.h"
#include <stdio.h>

template < typename T, char CWARP_SIZE >
__host__ void run_afl_compress_cpu( const unsigned int bit_length, T *data, T *compressed_data, const unsigned long length, unsigned long comprLength)
{

    const unsigned int block_size = CWARP_SIZE * 8;
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    unsigned int tid, bid;

    for (tid = 0, bid = 0; bid <= block_number; tid++)
    {
        if (tid == block_size)
        {
           tid = 0;
           bid += 1;
        }

        unsigned int warp_lane = (tid % CWARP_SIZE);
        unsigned long data_block = bid * block_size + tid - warp_lane;
        unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
        unsigned long cdata_id = data_block * bit_length + warp_lane;

        afl_compress_base_gpu <T, CWARP_SIZE> (bit_length, data_id, cdata_id, data, compressed_data, length, comprLength);
    }
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_compress_value_cpu( const unsigned int bit_length, T *data, T *compressed_data, const unsigned long length)
{

    unsigned long tid;

    for (tid = 0; tid < length; tid++)
        afl_compress_base_value_gpu <T, CWARP_SIZE> (bit_length, compressed_data, tid, data[tid]);
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_decompress_cpu(const unsigned int bit_length, T *compressed_data, T *decompress_data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8;
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    unsigned long tid, bid;

    for (tid = 0, bid = 0; bid < block_number; tid++)
    {
        if (tid == block_size)
        {
           tid = 0;
           bid += 1;
        }

        unsigned int warp_lane = (tid % CWARP_SIZE);
        unsigned long data_block = bid * block_size + tid - warp_lane;
        unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
        unsigned long cdata_id = data_block * bit_length + warp_lane;

        afl_decompress_base_gpu <T, CWARP_SIZE> (bit_length, cdata_id, data_id, compressed_data, decompress_data, length);
    }
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_compress_gpu(const unsigned int bit_length, T *data, T *compressed_data, unsigned long length, unsigned long comprLength)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    afl_compress_gpu <T, CWARP_SIZE> <<<block_number, block_size>>> (bit_length, data, compressed_data, length, comprLength);
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_decompress_gpu(const unsigned int bit_length, T *compressed_data, T *data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    afl_decompress_gpu <T, CWARP_SIZE> <<<block_number, block_size>>> (bit_length, compressed_data, data, length);
}

template < typename T, char CWARP_SIZE >
__host__ void run_afl_decompress_value_gpu(const unsigned int bit_length, T *compressed_data, T *data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size);
    afl_decompress_value_gpu <T, CWARP_SIZE> <<<block_number, block_size>>> (bit_length, compressed_data, data, length);
}

template < typename T, char CWARP_SIZE >
__global__ void afl_compress_gpu (const unsigned int bit_length, T *data, T *compressed_data, unsigned long length, unsigned long comprLength)
{
    const unsigned int warp_lane = (threadIdx.x % CWARP_SIZE);
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
    const unsigned long cdata_id = data_block * bit_length + warp_lane;

    afl_compress_base_gpu <T, CWARP_SIZE> (bit_length, data_id, cdata_id, data, compressed_data, length, comprLength);
}

template < typename T, char CWARP_SIZE >
__global__ void afl_decompress_gpu (const unsigned int bit_length, T *compressed_data, T * decompress_data, unsigned long length)
{
    const unsigned int warp_lane = (threadIdx.x % CWARP_SIZE);
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
    const unsigned long cdata_id = data_block * bit_length + warp_lane;

    afl_decompress_base_gpu <T, CWARP_SIZE> (bit_length, cdata_id, data_id, compressed_data, decompress_data, length);
}

template < typename T, char CWARP_SIZE >
__global__ void afl_decompress_value_gpu (const unsigned int bit_length, T *compressed_data, T * decompress_data, unsigned long length)
{
    const unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length)
    {
        decompress_data[tid] = afl_decompress_base_value_gpu <T, CWARP_SIZE> (bit_length, compressed_data, tid);
    }
}


template <typename T, char CWARP_SIZE>
__device__  __host__ void afl_compress_base_gpu (
		const unsigned int bit_length,
		unsigned long data_id,
		unsigned long comp_data_id,
		T *data,
		T *compressed_data,
		unsigned long length,
		unsigned long comprLength)
{
    T v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;

    for (unsigned int i = 0; (i < CWORD_SIZE(T)) && (pos_data < length); ++i)
    {
        v1 = data[pos_data];
        pos_data += CWARP_SIZE;

        if (v1_pos >= CWORD_SIZE(T) - bit_length && (pos < comprLength)){
            v1_len = CWORD_SIZE(T) - v1_pos;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);

            compressed_data[pos] = value;

            v1_pos = bit_length - v1_len;
            value = GETNPBITS(v1, v1_pos, v1_len);

            pos += CWARP_SIZE;
        } else {
            v1_len = bit_length;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);
            v1_pos += v1_len;
        }
    }
    if ((pos_data >= length)  && (pos_data < length + CWARP_SIZE) && (pos < comprLength))
    {
        compressed_data[pos] = value;
    }
}

template <typename T, char CWARP_SIZE>
__device__ __host__ void afl_decompress_base_gpu (const unsigned int bit_length, unsigned long comp_data_id, unsigned long data_id, T *compressed_data, T *data, unsigned long length)
{
    unsigned long pos = comp_data_id, pos_decomp = data_id;
    unsigned int v1_pos = 0, v1_len;
    T v1, ret;

    if (pos_decomp > length ) // Decompress not more elements then length
        return;
    v1 = compressed_data[pos];
    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_decomp < length; ++i)
    {
        if (v1_pos > CWORD_SIZE(T) - bit_length){
            v1_len = CWORD_SIZE(T) - v1_pos;
            ret = GETNPBITS(v1, v1_len, v1_pos);

            pos += CWARP_SIZE;
            v1 = compressed_data[pos];

            v1_pos = bit_length - v1_len;
            ret = ret | ((GETNBITS(v1, v1_pos))<< v1_len);
        } else {
            v1_len = bit_length;
            ret = GETNPBITS(v1, v1_len, v1_pos);
            v1_pos += v1_len;
        }

        data[pos_decomp] = ret;
        pos_decomp += CWARP_SIZE;
    }
}

template <typename T, char CWARP_SIZE>
__device__ __host__ T afl_decompress_base_value_gpu (
        const unsigned int bit_length,
        T *compressed_data,
        unsigned long pos
        )
{
    const unsigned int data_block = pos / (CWARP_SIZE * CWORD_SIZE(T));
    const unsigned int pos_in_block = (pos % (CWARP_SIZE * CWORD_SIZE(T)));
    const unsigned int pos_in_warp_lane = pos_in_block % CWARP_SIZE;
    const unsigned int pos_in_warp_comp_block = pos_in_block / CWARP_SIZE;

    const unsigned long cblock_id =
        data_block * ( CWARP_SIZE * bit_length)
        + pos_in_warp_lane
        + ((pos_in_warp_comp_block * bit_length) / CWORD_SIZE(T)) * CWARP_SIZE;

    const unsigned int bit_pos = pos_in_warp_comp_block * bit_length % CWORD_SIZE(T);
    const unsigned int bit_ret = bit_pos <= CWORD_SIZE(T)  - bit_length  ? bit_length : CWORD_SIZE(T) - bit_pos;

    T ret = GETNPBITS(compressed_data[cblock_id], bit_ret, bit_pos);

    if (bit_ret < bit_length)
        ret |= GETNBITS(compressed_data[cblock_id+CWARP_SIZE], bit_length - bit_ret) << bit_ret;

    return ret;
}

template <typename T, char CWARP_SIZE>
__device__ __host__ void afl_compress_base_value_gpu (
        const unsigned int bit_length,
        T *compressed_data,
        unsigned long pos,
        T value
        )
{
    const unsigned int data_block = pos / (CWARP_SIZE * CWORD_SIZE(T));
    const unsigned int pos_in_block = (pos % (CWARP_SIZE * CWORD_SIZE(T)));
    const unsigned int pos_in_warp_lane = pos_in_block % CWARP_SIZE;
    const unsigned int pos_in_warp_comp_block = pos_in_block / CWARP_SIZE;

    const unsigned long cblock_id =
        data_block * ( CWARP_SIZE * bit_length) // move to data block
        + pos_in_warp_lane // move to starting position in data block
        + ((pos_in_warp_comp_block * bit_length) / CWORD_SIZE(T)) * CWARP_SIZE; // move to value

    const unsigned int bit_pos = pos_in_warp_comp_block * bit_length % CWORD_SIZE(T);
    const unsigned int bit_ret = bit_pos <= CWORD_SIZE(T)  - bit_length  ? bit_length : CWORD_SIZE(T) - bit_pos;


    SETNPBITS(compressed_data + cblock_id, value, bit_ret, bit_pos);

    if (bit_ret < bit_length)
        SETNPBITS(compressed_data + cblock_id + CWARP_SIZE, (T)(value >> bit_ret), bit_length - bit_ret, 0);
}

// For now only those versions are available and will be compiled and linked
// This is intentional !!
#define GFL_SPEC(X, A) \
    template __host__ void run_afl_compress_gpu <X, A> (const unsigned int bit_length, X *data, X *compressed_data, unsigned long length, unsigned long);\
    template __host__ void run_afl_decompress_gpu <X, A> (const unsigned int bit_length, X *data, X *compressed_data, unsigned long length);\
    template __host__ void run_afl_compress_cpu <X, A> (const unsigned int bit_length, X *data, X *compressed_data, unsigned long length, unsigned long);\
    template __host__ void run_afl_decompress_cpu <X, A> (const unsigned int bit_length, X *data, X *compressed_data, unsigned long length);\
    template __host__ void run_afl_compress_value_cpu <X, A> (const unsigned int bit_length, X *data, X *compressed_data, unsigned long length);\
    template __host__ void run_afl_decompress_value_gpu <X, A> (const unsigned int bit_length, X *compressed_data, X *data, unsigned long length);

// A fast aligned version WARP_SIZE = 32
#define AFL_SPEC(X) GFL_SPEC(X, 32)
FOR_EACH(AFL_SPEC, char, short, int, long, unsigned int, unsigned long)

// Non aligned version - identical to classical CPU/GPU version (up to 10x slower then AFL)
#define FL_SPEC(X) GFL_SPEC(X, 1)
FOR_EACH(FL_SPEC, char, short, int, long, unsigned int, unsigned long)
