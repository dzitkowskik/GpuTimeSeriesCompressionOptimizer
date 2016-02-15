/*
 * gfc_encoding.cpp
 *
 *  Created on: Nov 17, 2015
 *      Author: Karol Dzitkowski
 */

#include "compression/gfc/gfc_encoding.hpp"
#include "compression/gfc/gfc_encoding_impl.cuh"
#include "compression/afl/afl_encoding.hpp"

namespace ddj
{

// DOUBLE
template<>
SharedCudaPtrVector<char> GfcEncoding::Encode(SharedCudaPtr<double> data)
{
	CUDA_ASSERT_RETURN( cudaGetLastError() );
    LOG4CPLUS_INFO_FMT(_logger, "GFC (DOUBLE) encoding START: data size = %lu", data->size());

	if(data->size() <= 0)
		return SharedCudaPtrVector<char>{
					CudaPtr<char>::make_shared(),
					CudaPtr<char>::make_shared(),
					CudaPtr<char>::make_shared()
					};

	unsigned long long size = data->size();
	int warpsperblock = 16;
	int subchunkCnt = 64;
	unsigned long long blocksize = warpsperblock * subchunkCnt * WARPSIZE;
	int blocks = (size + blocksize - 1) / blocksize;
	auto result = CompressDouble(data, blocks, warpsperblock);

	CUDA_ASSERT_RETURN( cudaGetLastError() );
    LOG4CPLUS_INFO(_logger, "GFC (DOUBLE) enoding END");

	return result;
}

template<>
SharedCudaPtr<double> GfcEncoding::Decode(SharedCudaPtrVector<char> input)
{
	LOG4CPLUS_INFO_FMT(
		_logger,
		"GFC (DOUBLE) decoding START: input[0] size = %lu, input[1] size = %lu, input[2] size = %lu",
		input[0]->size(), input[1]->size(), input[2]->size()
	);

	if(input[2]->size() <= 0)
		return CudaPtr<double>::make_shared();

	auto result = DecompressDouble(input);

	CUDA_ASSERT_RETURN( cudaGetLastError() );
    LOG4CPLUS_INFO(_logger, "GFC (DOUBLE) decoding END");

	return result;
}

// FLOAT
template<>
SharedCudaPtrVector<char> GfcEncoding::Encode(SharedCudaPtr<float> data)
{
	CUDA_ASSERT_RETURN( cudaGetLastError() );
    LOG4CPLUS_INFO_FMT(_logger, "GFC (FLOAT) encoding START: data size = %lu", data->size());

	if(data->size() <= 0)
		return SharedCudaPtrVector<char>{
					CudaPtr<char>::make_shared(),
					CudaPtr<char>::make_shared(),
					CudaPtr<char>::make_shared()
					};

	unsigned long long size = data->size();
	int warpsperblock = 32;
	int subchunkCnt = 32;
	unsigned long long blocksize = warpsperblock * subchunkCnt * WARPSIZE;
	int blocks = (size + blocksize - 1) / blocksize;
	auto result = CompressFloat(data, blocks, warpsperblock);

	CUDA_ASSERT_RETURN( cudaGetLastError() );
	size_t encodedSize = result[0]->size()+result[1]->size()+result[2]->size();
	LOG4CPLUS_INFO_FMT(_logger, "GFC (FLOAT) enoding (encoded to size %lu) END", encodedSize);

	return result;
}

template<>
SharedCudaPtr<float> GfcEncoding::Decode(SharedCudaPtrVector<char> input)
{
	LOG4CPLUS_INFO_FMT(
		_logger,
		"GFC (FLOAT) decoding START: input[0] size = %lu, input[1] size = %lu, input[2] size = %lu",
		input[0]->size(), input[1]->size(), input[2]->size()
	);

	if(input[2]->size() <= 0)
		return CudaPtr<float>::make_shared();

	auto result = DecompressFloat(input);

	CUDA_ASSERT_RETURN( cudaGetLastError() );
    LOG4CPLUS_INFO(_logger, "GFC (FLOAT) decoding END");

	return result;
}

SharedCudaPtrVector<char> GfcEncoding::EncodeFloat(SharedCudaPtr<float> data)
{ return this->Encode<float>(data); }
SharedCudaPtr<float> GfcEncoding::DecodeFloat(SharedCudaPtrVector<char> data)
{ return this->Decode<float>(data); }
SharedCudaPtrVector<char> GfcEncoding::EncodeDouble(SharedCudaPtr<double> data)
{ return this->Encode<double>(data); }
SharedCudaPtr<double> GfcEncoding::DecodeDouble(SharedCudaPtrVector<char> data)
{ return this->Decode<double>(data); }

SharedCudaPtrVector<char> GfcEncoding::EncodeInt(SharedCudaPtr<int> data)
{
	LOG4CPLUS_WARN(_logger, "GFC triggered on INT (using AFL)!!");
	auto result = AflEncoding().Encode<int>(data);
	result.push_back(CudaPtr<char>::make_shared());
	return result;
}
SharedCudaPtr<int> GfcEncoding::DecodeInt(SharedCudaPtrVector<char> data)
{
	LOG4CPLUS_WARN(_logger, "GFC triggered on INT (using AFL)!!");
	return AflEncoding().Decode<int>(data);
}
SharedCudaPtrVector<char> GfcEncoding::EncodeTime(SharedCudaPtr<time_t> data)
{
	LOG4CPLUS_WARN(_logger, "GFC triggered on TIME (using AFL)!!");
	auto result = AflEncoding().Encode<time_t>(data);
	result.push_back(CudaPtr<char>::make_shared());
	return result;
}
SharedCudaPtr<time_t> GfcEncoding::DecodeTime(SharedCudaPtrVector<char> data)
{
	LOG4CPLUS_WARN(_logger, "GFC triggered on TIME (using AFL)!!");
	return AflEncoding().Decode<time_t>(data);
}
SharedCudaPtrVector<char> GfcEncoding::EncodeShort(SharedCudaPtr<short> data)
{
	LOG4CPLUS_WARN(_logger, "GFC triggered on SHORT (using AFL)!!");
	auto result = AflEncoding().Encode<short>(data);
	result.push_back(CudaPtr<char>::make_shared());
	return result;
}
SharedCudaPtr<short> GfcEncoding::DecodeShort(SharedCudaPtrVector<char> data)
{
	LOG4CPLUS_WARN(_logger, "GFC triggered on SHORT (using AFL)!!");
	return AflEncoding().Decode<short>(data);
}
SharedCudaPtrVector<char> GfcEncoding::EncodeChar(SharedCudaPtr<char> data)
{
	LOG4CPLUS_WARN(_logger, "GFC triggered on CHAR (using AFL)!!");
	auto result = AflEncoding().Encode<char>(data);
	result.push_back(CudaPtr<char>::make_shared());
	return result;
}
SharedCudaPtr<char> GfcEncoding::DecodeChar(SharedCudaPtrVector<char> data)
{
	LOG4CPLUS_WARN(_logger, "GFC triggered on CHAR (using AFL)!!");
	return AflEncoding().Decode<char>(data);
}

} /* namespace ddj */
