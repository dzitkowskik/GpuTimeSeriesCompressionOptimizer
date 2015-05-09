/*
 * encode_decode_unittest_helper.cpp
 *
 *  Created on: 07-05-2015
 *      Author: ghash
 */

#include "encode_decode_unittest_helper.h"
#include "helpers/helper_comparison.cuh"
#include "helpers/helper_macros.h"
#include "helpers/helper_print.h"

namespace ddj
{

template<typename T>
bool EncodeDecodeUnittestHelper::TestSize(
		boost::function<SharedCudaPtr<char> (SharedCudaPtr<T> data)> encodeFunction,
		boost::function<SharedCudaPtr<T> (SharedCudaPtr<char> data)> decodeFunction,
		SharedCudaPtr<T> data)
{
	auto encodedData = encodeFunction(data);
	auto decodedData = decodeFunction(encodedData);
	return data->size() == decodedData->size();
}

template<typename T>
bool EncodeDecodeUnittestHelper::TestContent(
		boost::function<SharedCudaPtr<char> (SharedCudaPtr<T> data)> encodeFunction,
		boost::function<SharedCudaPtr<T> (SharedCudaPtr<char> data)> decodeFunction,
		SharedCudaPtr<T> data)
{
	auto encodedData = encodeFunction(data);
	auto decodedData = decodeFunction(encodedData);
	return CompareDeviceArrays(data->get(), decodedData->get(), data->size());
}

#define SCALE_SPEC(X) \
	template bool EncodeDecodeUnittestHelper::TestSize<X>(								\
			boost::function<SharedCudaPtr<char> (SharedCudaPtr<X> data)> encodeFunction,\
			boost::function<SharedCudaPtr<X> (SharedCudaPtr<char> data)> decodeFunction,\
			SharedCudaPtr<X> data); 													\
	template bool EncodeDecodeUnittestHelper::TestContent<X>(							\
			boost::function<SharedCudaPtr<char> (SharedCudaPtr<X> data)> encodeFunction,\
			boost::function<SharedCudaPtr<X> (SharedCudaPtr<char> data)> decodeFunction,\
			SharedCudaPtr<X> data);
FOR_EACH(SCALE_SPEC, double, float, int, long, long long, unsigned int, unsigned long, unsigned long long)

} /* namespace ddj */
