/*
 * gfc_encoding_impl_double.cu
 *
 *  Created on: Jan 9, 2016
 *      Author: Karol Dzitkowski
 */

#include "compression/gfc/gfc_encoding_impl.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>

__constant__ ull* 	cData;
__constant__ ull* 	cDecompressedData;
__constant__ uchar* cCompressedData;
__constant__ int* 	cChunkSizeData;
__constant__ int* 	cOffsetData;

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

__global__ void _gfcCompressKernelD()
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
	int offset = ((chunkBegin+1)/2*17);

	// loop through every subchunk in a chunk
	ull last = 0, diff = 0;
	for (int i = chunkBegin + warpIdx, sign, minBytes; i < chunkEnd; i += WARPSIZE)
	{
		// get a sign of a difference between data and last and diff as as absolute difference
		diff = cData[i] - last;
		sign = (diff >> 60) & 8;
		if(sign) diff = -diff;

		// get minimal number of bytes to store the diff
		minBytes = __clzll(diff) >> 3;
		// if there were 6 zero bytes we treat it like 5 zero bytes to store minBytes in 3 bits
		minBytes = minBytes == 6 ? 3 : 8 - minBytes;

		// count a prefix sum per warp in block from number of minBytes
		WarpPrefixSum(sBufferIdx, sBuffer, minBytes);

		int beg = offset + (WARPSIZE/2) + sBuffer[sBufferIdx-1];
		int end = beg + minBytes;
		for (; beg < end; beg++)
		{
			cCompressedData[beg] = diff;
			diff >>= 8;
		}

		if (minBytes >= 3) minBytes--;
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

__global__ void _gfcDecompressKernelD()
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
	int offset = ((chunkBegin+1)/2*17);

	ull diff = 0, last = 0;
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
		if (minBytes >= 2) minBytes++;

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

SharedCudaPtrVector<char> CompressDouble(SharedCudaPtr<double> data, int blocks, int warpsperblock)
{
	int doubles = data->size();
	int warpsCnt = blocks * warpsperblock;

	// calculate required padding for last chunk
	int padding = ((doubles + WARPSIZE - 1) & -WARPSIZE) - doubles;
	doubles += padding;

	SharedCudaPtr<double> uncompressed;
	if(padding > 0)
	{
		uncompressed = CudaPtr<double>::make_shared(doubles);
		uncompressed->set(0);
		uncompressed->fill(data->get(), data->size());
	}
	else
		uncompressed = data;

	// determine chunk assignments per warp
	int per = (doubles + blocks * warpsperblock - 1) / (blocks * warpsperblock);
	if (per < WARPSIZE) per = WARPSIZE;
	per = (per + WARPSIZE - 1) & -WARPSIZE;
	int curr = 0;
	int cut[warpsCnt];
	for (int i = 0; i < warpsCnt; i++)
	{
		curr += per;
		cut[i] = min(curr, doubles);
	}

	auto compressed = CudaPtr<char>::make_shared((doubles+1)/2*17);
	auto boundaries = CudaPtr<int>::make_shared(warpsCnt);
	auto offsets = CudaPtr<int>::make_shared(warpsCnt);

	auto uncompressedPtr = uncompressed->get();
	auto compressedPtr = compressed->get();
	auto boundariesPtr = boundaries->get();
	auto offsetsPtr = offsets->get();

	CUDA_CALL( cudaMemcpyToSymbol(cData, &uncompressedPtr, sizeof(void *)) );
	CUDA_CALL( cudaMemcpyToSymbol(cCompressedData, &compressedPtr, sizeof(void *)) );
	CUDA_CALL( cudaMemcpyToSymbol(cChunkSizeData, &boundariesPtr, sizeof(void *)) );
	CUDA_CALL( cudaMemcpyToSymbol(cOffsetData, &offsetsPtr, sizeof(void *)) );

	boundaries->fillFromHost(cut, warpsCnt);

	_gfcCompressKernelD<<<blocks, WARPSIZE*warpsperblock, 32 * (3 * WARPSIZE / 2) * sizeof(int)>>>();
	cudaDeviceSynchronize();
	CUDA_CALL( cudaGetLastError() );

	// CREATE METADATA
	auto metadata = CudaPtr<int>::make_shared(3);
	int h_metadata[3];
	h_metadata[0] = blocks;
	h_metadata[1] = warpsperblock;
	h_metadata[2] = doubles-padding;
	metadata->fillFromHost(h_metadata, 3);

	// get sizes of chunks
	auto h_offsets = offsets->copyToHost();
	int start = 0, totalSize = 0, total = 0, offset = 0;
	for(int i = 0; i < warpsCnt; i++)
	{
		if(i > 0) start = cut[i-1];
		(*h_offsets)[i] -= ((start+1)/2*17);
		totalSize += (*h_offsets)[i];
	}

	auto result = CudaPtr<char>::make_shared(totalSize);

	// copy compressed data by chunk
	start=0;
	for(int i = 0; i < blocks * warpsperblock; i++)
	{
		if(i > 0) start = cut[i-1];
		offset = ((start+1)/2*17);
		CUDA_CALL(
			cudaMemcpy(result->get()+total, compressed->get()+offset, (*h_offsets)[i], CPY_DTD)
		);
		total+=(*h_offsets)[i];
	}

	return SharedCudaPtrVector<char>{
		MoveSharedCudaPtr<int, char>(metadata),
		MoveSharedCudaPtr<int, char>(offsets),
		result
	};
}

SharedCudaPtr<double> DecompressDouble(SharedCudaPtrVector<char> input)
{
	auto metadata = input[0];
	auto offsets = CastSharedCudaPtr<char, int>(input[1]);
	auto data = input[2];

	int h_metadata[3];
	CUDA_CALL( cudaMemcpy(h_metadata, metadata->get(), metadata->size(), CPY_DTH) );
	int blocks = h_metadata[0];
	int warpsperblock = h_metadata[1];
	int doubles = h_metadata[2];
	int warpsCnt = blocks * warpsperblock;

	// calculate required padding for last chunk
	int padding = ((doubles + WARPSIZE - 1) & -WARPSIZE) - doubles;
	doubles += padding;

	// determine chunk assignments per warp
  	int cut[warpsCnt];
	int per = (doubles + blocks * warpsperblock - 1) / (blocks * warpsperblock);
	if (per < WARPSIZE) per = WARPSIZE;
	per = (per + WARPSIZE - 1) & -WARPSIZE;
	int curr = 0;
	for (int i = 0; i < blocks * warpsperblock; i++)
	{
		curr += per;
		cut[i] = min(curr, doubles);
	}

	auto h_offsets = offsets->copyToHost();

	auto compressed = CudaPtr<char>::make_shared((doubles+1)/2*17);
	auto decompressed = CudaPtr<double>::make_shared(doubles);
	auto boundaries = CudaPtr<int>::make_shared(warpsCnt);

	auto decompressedPtr = decompressed->get();
	auto compressedPtr = compressed->get();
	auto boundariesPtr = boundaries->get();

	CUDA_CALL( cudaMemcpyToSymbol(cDecompressedData, &decompressedPtr, sizeof(void *)) );
	CUDA_CALL( cudaMemcpyToSymbol(cCompressedData, &compressedPtr, sizeof(void *)) );
	CUDA_CALL( cudaMemcpyToSymbol(cChunkSizeData, &boundariesPtr, sizeof(void *)) );

    boundaries->fillFromHost(cut, warpsCnt);

	// copy compressed data by chunk
	int offset, start = 0, total=0, size=0;
	for(int i = 0; i < blocks * warpsperblock; i++)
	{
		if (i > 0) start = cut[i-1];
		offset = ((start+1)/2*17);
		size = (*h_offsets)[i]-offset;
		CUDA_CALL(cudaMemcpy(compressed->get() + offset, data->get() + total, size*sizeof(char), CPY_DTD));
		total += size;
	}

	_gfcDecompressKernelD<<<blocks, WARPSIZE*warpsperblock, 32 * (3 * WARPSIZE / 2) * sizeof(int)>>>();
	cudaDeviceSynchronize();
	CUDA_CALL( cudaGetLastError() );

	if(padding > 0)
		return decompressed->copy(doubles-padding);
	else return decompressed;
}
