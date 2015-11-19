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

template<>
SharedCudaPtrVector<char> GfcEncoding::Encode(SharedCudaPtr<double> data)
{
	unsigned long long size = data->size();
	int warpsperblock = 16;
	int subchunkCnt = 64;
	unsigned long long blocksize = warpsperblock * subchunkCnt * WARPSIZE;
	int blocks = (size + blocksize - 1) / blocksize;
	return CompressDouble(data, blocks, warpsperblock);
}

template<>
SharedCudaPtr<double> GfcEncoding::Decode(SharedCudaPtrVector<char> input)
{ return DecompressDouble(input); }

template<>
SharedCudaPtrVector<char> GfcEncoding::Encode(SharedCudaPtr<float> data)
{ return SharedCudaPtrVector<char>(); }

template<>
SharedCudaPtr<float> GfcEncoding::Decode(SharedCudaPtrVector<char> input)
{ return SharedCudaPtr<float>(); }

template<>
SharedCudaPtrVector<char> GfcEncoding::Encode(SharedCudaPtr<int> data)
{ return SharedCudaPtrVector<char>(); }

template<>
SharedCudaPtr<int> GfcEncoding::Decode(SharedCudaPtrVector<char> input)
{ return SharedCudaPtr<int>(); }

SharedCudaPtrVector<char> GfcEncoding::EncodeDouble(SharedCudaPtr<double> data)
{ return this->Encode<double>(data); }

SharedCudaPtr<double> GfcEncoding::DecodeDouble(SharedCudaPtrVector<char> data)
{ return this->Decode<double>(data); }

SharedCudaPtrVector<char> GfcEncoding::EncodeInt(SharedCudaPtr<int> data)
{ return this->Encode<int>(data); }

SharedCudaPtr<int> GfcEncoding::DecodeInt(SharedCudaPtrVector<char> data)
{ return this->Decode<int>(data); }

SharedCudaPtrVector<char> GfcEncoding::EncodeFloat(SharedCudaPtr<float> data)
{ return this->Encode<float>(data); }

SharedCudaPtr<float> GfcEncoding::DecodeFloat(SharedCudaPtrVector<char> data)
{ return this->Decode<float>(data); }


} /* namespace ddj */
