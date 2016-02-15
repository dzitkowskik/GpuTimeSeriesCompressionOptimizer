/*
 *  none_encoding.cpp
 *
 *  Created on: 15/10/2015
 *      Author: Karol Dzitkowski
 */

#include "compression/none/none_encoding.hpp"
#include "core/macros.h"

namespace ddj {

template<typename T>
SharedCudaPtrVector<char> NoneEncoding::Encode(SharedCudaPtr<T> data)
{
	CUDA_ASSERT_RETURN( cudaGetLastError() );
	LOG4CPLUS_INFO_FMT(_logger, "NONE encoding START: data size = %lu", data->size());

	// ALLOCATE RESULTS
	auto resultData = MoveSharedCudaPtr<T, char>(data->copy());
	auto resultMetadata = CudaPtr<char>::make_shared(sizeof(size_t));

	size_t dataSize = resultData->size();
	resultMetadata->fillFromHost((char*)&dataSize, sizeof(size_t));

	CUDA_ASSERT_RETURN( cudaGetLastError() );
    LOG4CPLUS_INFO(_logger, "NONE encoding END");

	return SharedCudaPtrVector<char> {resultMetadata, resultData};
}

template<typename T>
SharedCudaPtr<T> NoneEncoding::Decode(SharedCudaPtrVector<char> input)
{
	LOG4CPLUS_INFO_FMT(
		_logger,
		"NONE decoding START: input[0] size = %lu, input[1] size = %lu",
		input[0]->size(), input[1]->size()
	);

	auto data = input[1];
	auto resultData = data->copy();
	auto result = MoveSharedCudaPtr<char, T>(resultData);

	CUDA_ASSERT_RETURN( cudaGetLastError() );
	LOG4CPLUS_INFO(_logger, "RLE decoding END");

	return result;
}

#define NONE_ENCODING_SPEC(X) \
	template SharedCudaPtrVector<char> NoneEncoding::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> NoneEncoding::Decode<X>(SharedCudaPtrVector<char> data);
FOR_EACH(NONE_ENCODING_SPEC, char, short, double, float, int, long, long long, unsigned int)

} /* namespace ddj */
