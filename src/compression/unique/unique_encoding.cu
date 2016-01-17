/*
 * unique_encoding.cu
 *
 *  Created on: 8 paź 2015
 *      Author: Karol Dzitkowski
 */

#include "compression/unique/unique_encoding.hpp"
#include "core/cuda_macros.cuh"
#include "core/not_implemented_exception.hpp"
#include "core/cuda_launcher.cuh"
#include "helpers/helper_print.hpp"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace ddj {

SharedCudaPtr<char> UniqueEncoding::FindUnique(SharedCudaPtr<char> data, DataType type)
{
	switch(type)
	{
		case DataType::d_int:
			return boost::reinterpret_pointer_cast<CudaPtr<char>>(
					FindUnique(boost::reinterpret_pointer_cast<CudaPtr<int>>(data)));
		case DataType::d_float:
			return boost::reinterpret_pointer_cast<CudaPtr<char>>(
					FindUnique(boost::reinterpret_pointer_cast<CudaPtr<float>>(data)));
		default:
			throw NotImplementedException("No UniqueEncoding::FindUnique implementation for that type");
	}
}

template<typename T>
SharedCudaPtr<T> UniqueEncoding::FindUnique(SharedCudaPtr<T> data)
{
    thrust::device_ptr<T> dataPtr(data->get());
    thrust::device_vector<T> dataVector(dataPtr, dataPtr+data->size());
    thrust::sort(dataVector.begin(), dataVector.end());
    auto end = thrust::unique(dataVector.begin(), dataVector.end());
    int size = end - dataVector.begin();
    auto result = CudaPtr<T>::make_shared(size);
    result->fill(dataVector.data().get(), size);
    return result;
}

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
	if(data->size() <= 0)
		return CudaPtr<char>::make_shared();

    // CALCULATE SIZES
	int uniqueSize = unique->size();                        // how many distinct items to encode
	int dataSize = data->size();							// size of data to compress
    int bitsNeeded = ALT_BITLEN(uniqueSize-1);              // min bits needed to encode
    int outputItemBitSize = 8 * sizeof(unsigned int);      	// how many bits are in output unit
    int dataPerOutputCnt = outputItemBitSize / bitsNeeded;  // how many items will be encoded in single unit
    int outputSize = (dataSize + dataPerOutputCnt - 1) / dataPerOutputCnt;  // output units cnt
    int outputSizeInBytes = outputSize * sizeof(unsigned int);
    int uniqueSizeInBytes = unique->size() * sizeof(T);	// size in bytes of unique array
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
	size_t sizes[2];
	sizes[0] = uniqueSize;
	sizes[1] = dataSize;

    CUDA_CALL( cudaMemcpy(result->get(), &sizes, 2*sizeof(size_t), CPY_HTD) );
    CUDA_CALL( cudaMemcpy(result->get()+2*sizeof(size_t), unique->get(), uniqueSizeInBytes, CPY_DTD) );

	// printf("COMPRESS UNIQUE - uniqueSize = %d, dataSize = %d\n", uniqueSize, dataSize);

	cudaDeviceSynchronize();
	CUDA_ASSERT_RETURN( cudaGetLastError() );
    return result;
}

template<typename T>
SharedCudaPtrVector<char> UniqueEncoding::Encode(SharedCudaPtr<T> data)
{
	if(data->size() <= 0)
		return SharedCudaPtrVector<char>{ CudaPtr<char>::make_shared(), CudaPtr<char>::make_shared() };

	//	HelperPrint::PrintSharedCudaPtr(data, "After Delta");

    auto unique = FindUnique(data);

    // CALCULATE SIZES
	int uniqueSize = unique->size();                        // how many distinct items to encode
	int dataSize = data->size();							// size of data to compress
    int bitsNeeded = ALT_BITLEN(uniqueSize-1);              // min bits needed to encode
    int outputItemBitSize = 8 * sizeof(unsigned int);      	// how many bits are in output unit
    int dataPerOutputCnt = outputItemBitSize / bitsNeeded;  // how many items will be encoded in single unit
    int outputSize = (dataSize + dataPerOutputCnt - 1) / dataPerOutputCnt;  // output units cnt
    int outputSizeInBytes = outputSize * sizeof(unsigned int);

    // COMPRESS UNIQUE
    auto resultData = CudaPtr<char>::make_shared(outputSizeInBytes);
    this->_policy.setSize(outputSize);
    cudaLaunch(this->_policy, _compressUniqueKernel<T>,
        data->get(),
        data->size(),
        bitsNeeded,
        dataPerOutputCnt,
        unique->get(),
        unique->size(),
        (unsigned int*)resultData->get(),
        outputSize);
    cudaDeviceSynchronize();

    size_t metadataSize = unique->size() * sizeof(T) + 2*sizeof(size_t);
    auto resultMetadata = CudaPtr<char>::make_shared(metadataSize);
    size_t sizes[2] {unique->size(), data->size()};
    CUDA_CALL( cudaMemcpy(resultMetadata->get(), sizes, 2*sizeof(size_t), CPY_HTD) );
    CUDA_CALL( cudaMemcpy(
        resultMetadata->get()+2*sizeof(size_t),
        unique->get(),
        unique->size()*sizeof(T),
        CPY_DTD) );
    return SharedCudaPtrVector<char> {resultMetadata, resultData};
}

template<typename T>
__global__ void _decompressUniqueKernel(
    unsigned int* data,
    T* unique,
    const int size,
    T* output,
    const int outputSize,
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
        int outputIndex = idx * dataPerUnitCnt + i;
        if(outputIndex < outputSize)
            output[outputIndex] = value;
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
	if(data->size() <= 0)
		return CudaPtr<T>::make_shared();

	// GET SIZES
	size_t sizes[2];
	CUDA_CALL( cudaMemcpy(&sizes[0], data->get(), 2*sizeof(size_t), CPY_DTH) );
	int uniqueCnt = sizes[0];
	int outputSize = sizes[1];

	// printf("UNIQUE ENCODING - uniqueCnt = %d\n", uniqueCnt);
	// printf("UNIQUE ENCODING - outputSize = %d\n", outputSize);

	// GETE UNIQUE VALUES DATA
	auto unique = CudaPtr<T>::make_shared(uniqueCnt);
	T* uniqueDataPtr = (T*)(data->get()+2*sizeof(size_t));
	unique->fill(uniqueDataPtr, uniqueCnt);

	// CALCULATE SIZES
    int bitsNeeded = ALT_BITLEN(uniqueCnt-1);          		// min bit cnt to encode unique values
    int unitSize = sizeof(unsigned int);                    // single unit size in bytes
    int unitBitSize = 8 * sizeof(unsigned int);             // single unit size in bits
    int dataPerUnitCnt = unitBitSize / bitsNeeded;          // how many items are in one unit
    int unitCnt =  data->size() / unitSize;                 // how many units are in data
    int uniqueSizeInBytes = uniqueCnt * sizeof(T);    		// size in bytes of unique array
    int headerSize = uniqueSizeInBytes + 2 * sizeof(size_t);// size of data header

    // DECOMPRESS DATA USING UNIQUE VALUES
    auto result = CudaPtr<T>::make_shared(outputSize);
    this->_policy.setSize(unitCnt);
    cudaLaunch(this->_policy, _decompressUniqueKernel<T>,
        (unsigned int*)(data->get()+headerSize),
        (T*)(data->get()+2*sizeof(size_t)),
        unitCnt,
        result->get(),
        result->size(),
        bitsNeeded,
        dataPerUnitCnt);
    cudaDeviceSynchronize();
	CUDA_ASSERT_RETURN( cudaGetLastError() );

    return result;
}

template<typename T>
SharedCudaPtr<T> UniqueEncoding::Decode(SharedCudaPtrVector<char> input)
{
	if(input[1]->size() <= 0)
		return CudaPtr<T>::make_shared();

    auto metadata = input[0];
    auto data = input[1];
    size_t sizes[2];
    CUDA_CALL( cudaMemcpy(sizes, metadata->get(), 2*sizeof(size_t), CPY_DTH) );
    size_t uniqueSize = sizes[0];
    size_t outputSize = sizes[1];

    // CALCULATE SIZES
    int bitsNeeded = ALT_BITLEN(uniqueSize-1);          	// min bit cnt to encode unique values
    int unitSize = sizeof(unsigned int);                    // single unit size in bytes
    int unitBitSize = 8 * sizeof(unsigned int);             // single unit size in bits
    int dataPerUnitCnt = unitBitSize / bitsNeeded;          // how many items are in one unit
    int unitCnt =  data->size() / unitSize;                 // how many units are in data

    // DECOMPRESS DATA USING UNIQUE VALUES
    auto result = CudaPtr<T>::make_shared(outputSize);
    this->_policy.setSize(unitCnt);
    cudaLaunch(this->_policy, _decompressUniqueKernel<T>,
        (unsigned int*)data->get(),
        (T*)(metadata->get()+2*sizeof(size_t)),
        unitCnt,
        result->get(),
        result->size(),
        bitsNeeded,
        dataPerUnitCnt);

    cudaDeviceSynchronize();

//    HelperPrint::PrintSharedCudaPtr(result, "decoded by unique");

    return result;
}

size_t UniqueEncoding::GetMetadataSize(SharedCudaPtr<char> data, DataType type)
{
	if(data->size() <= 0) return 0;
	auto unique = FindUnique(data, type);
	return 2*sizeof(size_t) + unique->size();
}

size_t UniqueEncoding::GetCompressedSize(SharedCudaPtr<char> data, DataType type)
{
	if(data->size() <= 0) return 0;
	auto unique = FindUnique(data, type);
	int typeSize = GetDataTypeSize(type); 					// size of used type
	int uniqueSize = unique->size() / typeSize; 			// how many distinct items to encode
	int dataSize = data->size() / typeSize; 				// no elements to compress
	int bitsNeeded = ALT_BITLEN(uniqueSize-1); 				// min bits needed to encode
	int outputItemBitSize = 8 * sizeof(unsigned int);      	// how many bits are in output unit
	int dataPerOutputCnt = outputItemBitSize / bitsNeeded;  // how many items will be encoded in single unit
	int outputSize = (dataSize + dataPerOutputCnt - 1) / dataPerOutputCnt;  // output units cnt
	int outputSizeInBytes = outputSize * sizeof(unsigned int);	// output size after compression
	return outputSizeInBytes;
}


#define UNIQUE_ENCODING_SPEC(X) \
	template SharedCudaPtr<char> UniqueEncoding::CompressUnique<X>(SharedCudaPtr<X>, SharedCudaPtr<X>); \
	template SharedCudaPtr<X> UniqueEncoding::DecompressUnique<X>(SharedCudaPtr<char>); \
    template SharedCudaPtrVector<char> UniqueEncoding::Encode<X>(SharedCudaPtr<X>); \
    template SharedCudaPtr<X> UniqueEncoding::Decode<X>(SharedCudaPtrVector<char>);
FOR_EACH(UNIQUE_ENCODING_SPEC, char, short, double, float, int, long, long long, unsigned int)

} /* namespace ddj */
