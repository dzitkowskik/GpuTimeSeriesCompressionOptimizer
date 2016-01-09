/*
 * gfc_encoding.cpp
 *
 *  Created on: Nov 17, 2015
 *      Author: Karol Dzitkowski
 */

#include "compression/gfc/gfc_encoding.hpp"
#include "compression/gfc/gfc_encoding_impl.cuh"

namespace ddj
{

// DOUBLE
template<>
SharedCudaPtrVector<char> GfcEncoding::Encode(SharedCudaPtr<double> data)
{
	if(data->size() <= 0)
		return SharedCudaPtrVector<char>{ CudaPtr<char>::make_shared(), CudaPtr<char>::make_shared() };


	unsigned long long size = data->size();
	int warpsperblock = 16;
	int subchunkCnt = 64;
	unsigned long long blocksize = warpsperblock * subchunkCnt * WARPSIZE;
	int blocks = (size + blocksize - 1) / blocksize;
	return CompressDouble(data, blocks, warpsperblock);
}

template<>
SharedCudaPtr<double> GfcEncoding::Decode(SharedCudaPtrVector<char> input)
{
	if(input[1]->size() <= 0)
		return CudaPtr<double>::make_shared();

	return DecompressDouble(input);
}

// FLOAT
template<>
SharedCudaPtrVector<char> GfcEncoding::Encode(SharedCudaPtr<float> data)
{
	if(data->size() <= 0)
		return SharedCudaPtrVector<char>{ CudaPtr<char>::make_shared(), CudaPtr<char>::make_shared() };

	unsigned long long size = data->size();
	int warpsperblock = 32;
	int subchunkCnt = 32;
	unsigned long long blocksize = warpsperblock * subchunkCnt * WARPSIZE;
	int blocks = (size + blocksize - 1) / blocksize;
	return CompressFloat(data, blocks, warpsperblock);
}

template<>
SharedCudaPtr<float> GfcEncoding::Decode(SharedCudaPtrVector<char> input)
{
	if(input[1]->size() <= 0)
		return CudaPtr<float>::make_shared();

	return DecompressFloat(input);
}

// INT
template<>
SharedCudaPtrVector<char> GfcEncoding::Encode(SharedCudaPtr<int> data)
{ return SharedCudaPtrVector<char>(); }
template<>
SharedCudaPtr<int> GfcEncoding::Decode(SharedCudaPtrVector<char> input)
{ return SharedCudaPtr<int>(); }

// TIME
template<>
SharedCudaPtrVector<char> GfcEncoding::Encode(SharedCudaPtr<time_t> data)
{ return SharedCudaPtrVector<char>(); }
template<>
SharedCudaPtr<time_t> GfcEncoding::Decode(SharedCudaPtrVector<char> input)
{ return SharedCudaPtr<time_t>(); }


SharedCudaPtrVector<char> GfcEncoding::EncodeInt(SharedCudaPtr<int> data)
{ return this->Encode<int>(data); }
SharedCudaPtr<int> GfcEncoding::DecodeInt(SharedCudaPtrVector<char> data)
{ return this->Decode<int>(data); }
SharedCudaPtrVector<char> GfcEncoding::EncodeTime(SharedCudaPtr<time_t> data)
{ return this->Encode<time_t>(data); }
SharedCudaPtr<time_t> GfcEncoding::DecodeTime(SharedCudaPtrVector<char> data)
{ return this->Decode<time_t>(data); }
SharedCudaPtrVector<char> GfcEncoding::EncodeFloat(SharedCudaPtr<float> data)
{ return this->Encode<float>(data); }
SharedCudaPtr<float> GfcEncoding::DecodeFloat(SharedCudaPtrVector<char> data)
{ return this->Decode<float>(data); }
SharedCudaPtrVector<char> GfcEncoding::EncodeDouble(SharedCudaPtr<double> data)
{ return SharedCudaPtrVector<char>(); }
SharedCudaPtr<double> GfcEncoding::DecodeDouble(SharedCudaPtrVector<char> data)
{ return SharedCudaPtr<double>(); }

} /* namespace ddj */
