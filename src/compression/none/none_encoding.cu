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

	// ALLOCATE RESULTS
	auto resultData = MoveSharedCudaPtr<T, char>(data->copy());
	auto resultMetadata = CudaPtr<char>::make_shared(sizeof(size_t));

	size_t dataSize = resultData->size();
	resultMetadata->fillFromHost((char*)&dataSize, sizeof(size_t));

	return SharedCudaPtrVector<char> {resultMetadata, resultData};
}

template<typename T>
SharedCudaPtr<T> NoneEncoding::Decode(SharedCudaPtrVector<char> input)
{
	// printf("Decode using NoneEncoding - metadata size = %lu, data size = %lu\n", input[0]->size(),input[1]->size());
	auto data = input[1];
	auto resultData = data->copy();
	return MoveSharedCudaPtr<char, T>(resultData);
}

#define NONE_ENCODING_SPEC(X) \
	template SharedCudaPtrVector<char> NoneEncoding::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> NoneEncoding::Decode<X>(SharedCudaPtrVector<char> data);
FOR_EACH(NONE_ENCODING_SPEC, char, short, double, float, int, long, long long, unsigned int)

} /* namespace ddj */
