/*
 * gfc_encoding_impl_float.cu
 *
 *  Created on: Jan 9, 2016
 *      Author: Karol Dzitkowski
 */

#include "compression/gfc/gfc_encoding_impl.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>

template<typename T>
__device__ void WarpPrefixSum(int idx, T* arr, T value)
{
	arr[idx] = value; 			__threadfence_block();
	arr[idx] += arr[idx-1]; 	__threadfence_block();
	arr[idx] += arr[idx-2]; 	__threadfence_block();
	arr[idx] += arr[idx-4]; 	__threadfence_block();
	arr[idx] += arr[idx-8]; 	__threadfence_block();
	arr[idx] += arr[idx-16];	__threadfence_block();
}

__global__ void _gfcCompressKernelF(
		unsigned int* 	cData,
		uchar* cCompressedData,
		int* 	cChunkSizeData,
		int* 	cOffsetData)
{
	extern __shared__ int sBuffer[];

	// get indexes
	int warp = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;
	int warpIdx = threadIdx.x % WARPSIZE;
	int sBufferIdx = threadIdx.x / WARPSIZE * (3 * WARPSIZE / 2) + warpIdx;
	int sBufferLastIdx = (threadIdx.x / WARPSIZE + 1) * (3 * WARPSIZE / 2) - 1;

	// zero buffer and add half of WARP to index (for prefix sum)
	sBuffer[sBufferIdx] = 0;
	sBufferIdx += WARPSIZE / 2;

	// set begin and end of a chunk
	int chunkBegin = 0;
	if (warp > 0) chunkBegin = cChunkSizeData[warp-1];
	int chunkEnd = cChunkSizeData[warp];
	int offset = ((chunkBegin+1)/2*9);

	// loop through every subchunk in a chunk
	unsigned int last = 0, diff = 0;
	for (int i = chunkBegin + warpIdx, sign, minBytes; i < chunkEnd; i += WARPSIZE)
	{
		// get a sign of a difference between data and last and diff as as absolute difference
		diff = cData[i] - last;
		sign = (diff >> 28) & 8;
		if(sign) diff = -diff;

		// get minimal number of bytes to store the diff
		minBytes = __clz(diff) >> 3;
		minBytes = 4 - minBytes;

		// count a prefix sum per warp in block from number of minBytes
		WarpPrefixSum(sBufferIdx, sBuffer, minBytes);

		int beg = offset + (WARPSIZE/2) + sBuffer[sBufferIdx-1];
		int end = beg + minBytes;
		for (; beg < end; beg++)
		{
			cCompressedData[beg] = diff;
			diff >>= 8;
		}

		int tmp = sBuffer[sBufferLastIdx];
		sign |= minBytes;
		sBuffer[sBufferIdx] = sign;
		__threadfence_block();

		if ((warpIdx & 1) != 0)
			cCompressedData[offset + (warpIdx >> 1)] = sBuffer[sBufferIdx-1] | (sign << 4);

		offset += tmp + (WARPSIZE/2);
		last = cData[chunkBegin + 31];
	}

	if (warpIdx == 31) cOffsetData[warp] = offset;
}

__global__ void _gfcDecompressKernelF(
		uchar* cCompressedData,
		unsigned int* 	cDecompressedData,
		int* 	cChunkSizeData)
{
	extern __shared__ int sBuffer[];

	// get indexes
	int warp = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;
	int warpIdx = threadIdx.x % WARPSIZE;
	int sBufferIdx = threadIdx.x / WARPSIZE * (3 * WARPSIZE / 2) + warpIdx;
	int sBufferLastIdx = (threadIdx.x / WARPSIZE + 1) * (3 * WARPSIZE / 2) - 1;

	// zero buffer and add half of WARP to index (for prefix sum)
	sBuffer[sBufferIdx] = 0;
	sBufferIdx += WARPSIZE / 2;

	// set begin and end of a chunk
	int chunkBegin = 0;
	if (warp > 0) chunkBegin = cChunkSizeData[warp-1];
	int chunkEnd = cChunkSizeData[warp];
	int offset = ((chunkBegin+1)/2*9);

	unsigned int diff = 0, last = 0;
	for (int i = chunkBegin + warpIdx, sign, minBytes; i < chunkEnd; i += WARPSIZE)
	{
		// read in half-bytes of size and leading-zero count information
		if ((warpIdx & 1) == 0)
		{
			sign = cCompressedData[offset + (warpIdx >> 1)];
			sBuffer[sBufferIdx] = sign;
			sBuffer[sBufferIdx + 1] = sign >> 4;
		}
		offset += (WARPSIZE/2);
		__threadfence_block();
		sign = sBuffer[sBufferIdx];

		minBytes = sign & 7;
		WarpPrefixSum(sBufferIdx, sBuffer, minBytes);

		// read in compressed data (the non-zero bytes)
		int beg = offset + sBuffer[sBufferIdx-1];
		offset += sBuffer[sBufferLastIdx];
		int end = beg + minBytes - 1;
		diff = 0;
		for (; beg <= end; end--)
		{
			diff <<= 8;
			diff |= cCompressedData[end];
		}

		if ((sign & 8) != 0) diff = -diff;
		cDecompressedData[i] = last + diff;
		__threadfence_block();

		last = cDecompressedData[chunkBegin + 31];
	}
}

SharedCudaPtrVector<char> CompressFloat(SharedCudaPtr<float> data, int blocks, int warpsperblock)
{
	int floats = data->size();
	int warpsCnt = blocks * warpsperblock;

	// calculate required padding for last chunk
	int padding = ((floats + WARPSIZE - 1) & -WARPSIZE) - floats;
	floats += padding;

	SharedCudaPtr<float> uncompressed;
	if(padding > 0)
	{
		uncompressed = CudaPtr<float>::make_shared(floats);
		uncompressed->set(0);
		uncompressed->fill(data->get(), data->size());
	}
	else
		uncompressed = data;

	// determine chunk assignments per warp
	int per = (floats + blocks * warpsperblock - 1) / (blocks * warpsperblock);
	if (per < WARPSIZE) per = WARPSIZE;
	per = (per + WARPSIZE - 1) & -WARPSIZE;
	int curr = 0;
	int cut[warpsCnt];
	for (int i = 0; i < warpsCnt; i++)
	{
		curr += per;
		cut[i] = min(curr, floats);
	}

	auto compressed = CudaPtr<char>::make_shared((floats+1)/2*9);
	auto boundaries = CudaPtr<int>::make_shared(warpsCnt);
	auto offsets = CudaPtr<int>::make_shared(warpsCnt);
	boundaries->fillFromHost(cut, warpsCnt);

	_gfcCompressKernelF<<<blocks, WARPSIZE*warpsperblock, 32 * (3 * WARPSIZE / 2) * sizeof(int)>>>(
			(unsigned int*)uncompressed->get(),
			(uchar*)compressed->get(),
			boundaries->get(),
			offsets->get());
	cudaDeviceSynchronize();
	CUDA_CALL( cudaGetLastError() );

	// CREATE METADATA
	auto metadata = CudaPtr<int>::make_shared(3);
	int h_metadata[3];
	h_metadata[0] = blocks;
	h_metadata[1] = warpsperblock;
	h_metadata[2] = floats-padding;
	metadata->fillFromHost(h_metadata, 3);

	// get sizes of chunks
	auto h_offsets = offsets->copyToHost();
	int start = 0, totalSize = 0, total = 0, offset = 0;
	for(int i = 0; i < warpsCnt; i++)
	{
		if(i > 0) start = cut[i-1];
		(*h_offsets)[i] -= ((start+1)/2*9);
		totalSize += (*h_offsets)[i];
	}

	auto result = CudaPtr<char>::make_shared(totalSize);

	// copy compressed data by chunk
	start=0;
	for(int i = 0; i < blocks * warpsperblock; i++)
	{
		if(i > 0) start = cut[i-1];
		offset = ((start+1)/2*9);
		CUDA_CALL(
			cudaMemcpy(result->get()+total, compressed->get()+offset, (*h_offsets)[i], CPY_DTD)
		);
//		printf("result_size=%lu, total=%d, offset=%d, size=%d\n", result->size(), total, offset, (*h_offsets)[i]);
		total+=(*h_offsets)[i];
	}

	return SharedCudaPtrVector<char>{
		MoveSharedCudaPtr<int, char>(metadata),
		MoveSharedCudaPtr<int, char>(offsets),
		result
	};
}

SharedCudaPtr<float> DecompressFloat(SharedCudaPtrVector<char> input)
{
	auto metadata = input[0];
	auto offsets = CastSharedCudaPtr<char, int>(input[1]);
	auto data = input[2];

	int h_metadata[3];
	CUDA_CALL( cudaMemcpy(h_metadata, metadata->get(), metadata->size(), CPY_DTH) );
	int blocks = h_metadata[0];
	int warpsperblock = h_metadata[1];
	int floats = h_metadata[2];
	int warpsCnt = blocks * warpsperblock;

	// calculate required padding for last chunk
	int padding = ((floats + WARPSIZE - 1) & -WARPSIZE) - floats;
	floats += padding;

	// determine chunk assignments per warp
  	int cut[warpsCnt];
	int per = (floats + blocks * warpsperblock - 1) / (blocks * warpsperblock);
	if (per < WARPSIZE) per = WARPSIZE;
	per = (per + WARPSIZE - 1) & -WARPSIZE;
	int curr = 0;
	for (int i = 0; i < blocks * warpsperblock; i++)
	{
		curr += per;
		cut[i] = min(curr, floats);
	}

	auto h_offsets = offsets->copyToHost();

	auto compressed = CudaPtr<char>::make_shared((floats+1)/2*9);
	auto decompressed = CudaPtr<float>::make_shared(floats);
	auto boundaries = CudaPtr<int>::make_shared(warpsCnt);
    boundaries->fillFromHost(cut, warpsCnt);

	// copy compressed data by chunk
	int offset, start = 0, total=0, size=0;
	for(int i = 0; i < blocks * warpsperblock; i++)
	{
		if (i > 0) start = cut[i-1];
		offset = ((start+1)/2*9);
		size = (*h_offsets)[i]-offset;
		CUDA_CALL(cudaMemcpy(compressed->get() + offset, data->get() + total, size*sizeof(char), CPY_DTD));
//		printf("cmpr_size=%lu, data_size=%lu, offset=%d, total=%d, size=%d\n", compressed->size(), data->size(), offset, total, size);
		total += size;
	}

	_gfcDecompressKernelF<<<blocks, WARPSIZE*warpsperblock, 32 * (3 * WARPSIZE / 2) * sizeof(int)>>>(
			(uchar*)compressed->get(),
			(unsigned int*)decompressed->get(),
			boundaries->get());
	cudaDeviceSynchronize();
	CUDA_CALL( cudaGetLastError() );

	if(padding > 0)
		return decompressed->copy(floats-padding);
	else return decompressed;
}

