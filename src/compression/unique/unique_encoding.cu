/*
 * unique_encoding.cu
 *
 *  Created on: 8 paź 2015
 *      Author: Karol Dzitkowski
 */

#include "compression/unique/unique_encoding.hpp"
#include "core/cuda_macros.cuh"
#include "helpers/helper_cuda.cuh"

namespace ddj {


template<typename T>
__global__ void _compressUniqueKernel(
    T* data,
    int dataSize,
    int bitsNeeded,
    int dataPerOutputCnt,
    T* uniqueValues,
    int freqCnt,
    unsigned int* output,
    int outputSize)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // output index
    if(idx >= outputSize) return;

    unsigned int result = 0;

    for(int i = 0; i < dataPerOutputCnt; i++)
    {
        T value = data[idx * dataPerOutputCnt + i];
        for(int j = 0; j < freqCnt; j++)
        {
            if(value == uniqueValues[j])
            {
                result = SaveNbitIntValToWord(bitsNeeded, i, j, result);
            }
        }
    }
    output[idx] = result;
}


// UNIQUE COMPRESSION ALGORITHM
//
// 1. As the first part we save size of unique values array and size of data to compress
// 2. Next we store an array of unique values
// 2. Then we compress all data (containing these unique values) using N bits, where
// 	N is the smallest number of bits we can use to encode unique values array length - 1
// 	We encode this way each value as it's index in unique values array. We store data in an
// 	array of unsigned int values, putting as many values as possible in unsigned int variables.
// 	For example if we need 4 bits to encode single value, we put 8 values in one unsigned int
// 	and store it in output table.
template<typename T>
SharedCudaPtr<char> UniqueEncoding::CompressUnique(SharedCudaPtr<T> data, SharedCudaPtr<T> unique)
{
    // CALCULATE SIZES
	int uniqueSize = unique->size();                        // how many distinct items to encode
	int dataSize = data->size();							// size of data to compress
    int bitsNeeded = ALT_BITLEN(uniqueSize-1);              // min bits needed to encode
    int outputItemBitSize = 8 * sizeof(unsigned int);      	// how many bits are in output unit
    int dataPerOutputCnt = outputItemBitSize / bitsNeeded;  // how many items will be encoded in single unit
    int outputSize = (dataSize + dataPerOutputCnt - 1) / dataPerOutputCnt;  // output units cnt
    int outputSizeInBytes = outputSize * sizeof(unsigned int);
    int uniqueSizeInBytes = unique->size() * sizeof(int);	// size in bytes of unique array
    int headerSize = uniqueSizeInBytes + 2 * sizeof(size_t);// size of data header

    // COMPRESS UNIQUE
    auto result = CudaPtr<char>::make_shared(outputSizeInBytes + headerSize);
    this->_policy.setSize(outputSize);
    cudaLaunch(this->_policy, _compressUniqueKernel<T>,
        data->get(),
        data->size(),
        bitsNeeded,
        dataPerOutputCnt,
        unique->get(),
        unique->size(),
        (unsigned int*)(result->get()+headerSize),
        outputSize);

    // ATTACH HEADER
    CUDA_CALL( cudaMemcpy(result->get(), &uniqueSize, sizeof(size_t), CPY_HTD) );
    CUDA_CALL( cudaMemcpy(result->get()+sizeof(size_t), &dataSize, sizeof(size_t), CPY_HTD) );
    CUDA_CALL( cudaMemcpy(result->get()+2*sizeof(size_t), unique->get(), uniqueSizeInBytes, CPY_DTD) );
    cudaDeviceSynchronize();

    return result;
}


template<typename T>
__global__ void _decompressUniqueKernel(
    unsigned int* data,
    T* unique,
    const int size,
    T* output,
    const int bitsNeeded,
    const int dataPerUnitCnt
    )
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // output index
    if(idx >= size) return;
    for(int i=0; i<dataPerUnitCnt; i++)
    {
        int index = ReadNbitIntValFromWord(bitsNeeded, i, data[idx]);
        T value = unique[index];
        output[idx * dataPerUnitCnt + i] = value;
    }
}


// MOST FREQUENT DECOMPRESSION ALGORITHM
//
// 1. First we read number of unique values which are the first 4 bytes of data
// 2. Having the number of unique values we restore the original array of unique values
// 3. Then we count how many unsigned int values fit in encoded data and what is the minimal bit
// 	number to encode the number of unique values. Using that we can compute how many values
// 	are stored in sigle unsigned int number.
// 4. Then we take in parallel each unsigned int and decode it as the values from the array
// 	at index equal to that N bit block stored in unsigned int number casted to integer.
template<typename T>
SharedCudaPtr<T> UniqueEncoding::DecompressUnique(SharedCudaPtr<char> data)
{
	// GET SIZES
	size_t sizes[2];
	CUDA_CALL( cudaMemcpy(&sizes[0], data->get(), 2*sizeof(size_t), CPY_DTH) );
	int uniqueCnt = sizes[0];
	int outputSize = sizes[1];

	// GETE UNIQUE VALUES DATA
	auto unique = CudaPtr<T>::make_shared(uniqueCnt);
	T* uniqueDataPtr = (T*)(data->get()+sizeof(size_t));
	unique->fill(uniqueDataPtr, uniqueCnt);

	// CALCULATE SIZES
    int bitsNeeded = ALT_BITLEN(uniqueCnt-1);          		// min bit cnt to encode unique values
    int unitSize = sizeof(unsigned int);                    // single unit size in bytes
    int unitBitSize = 8 * sizeof(unsigned int);             // single unit size in bits
    int dataPerUnitCnt = unitBitSize / bitsNeeded;          // how many items are in one unit
    int unitCnt =  data->size() / unitSize;                 // how many units are in data
    int uniqueSizeInBytes = uniqueCnt * sizeof(int);    	// size in bytes of unique array
    int headerSize = uniqueSizeInBytes + 2 * sizeof(size_t);// size of data header

    // DECOMPRESS DATA USING UNIQUE VALUES
    auto result = CudaPtr<T>::make_shared(outputSize);
    this->_policy.setSize(unitCnt);
    cudaLaunch(this->_policy, _decompressUniqueKernel<T>,
        (unsigned int*)(data->get()+headerSize),
        (T*)(data->get()+2*sizeof(size_t)),
        unitCnt,
        result->get(),
        bitsNeeded,
        dataPerUnitCnt);

    cudaDeviceSynchronize();
    return result;
}


#define UNIQUE_ENCODING_SPEC(X) \
	template SharedCudaPtr<char> UniqueEncoding::CompressUnique<X>(SharedCudaPtr<X>, SharedCudaPtr<X>); \
	template SharedCudaPtr<X> UniqueEncoding::DecompressUnique<X>(SharedCudaPtr<char>);
FOR_EACH(UNIQUE_ENCODING_SPEC, float, int, long long, unsigned int)

} /* namespace ddj */
