/*
 * encode_decode_unittest_helper.cpp
 *
 *  Created on: 07-05-2015
 *      Author: Karol Dzitkowski
 */

#include "encode_decode_unittest_helper.hpp"
#include "helpers/helper_comparison.cuh"
#include "helpers/helper_macros.h"
#include "helpers/helper_print.hpp"

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

//	HelperPrint::PrintTestArrays<T>(data->get(), decodedData->get(), data->size());

	return CompareDeviceArrays(data->get(), decodedData->get(), data->size());
}

template<typename T>
bool EncodeDecodeUnittestHelper::TestSize2(
		boost::function<SharedCudaPtrVector<char> (SharedCudaPtr<T> data)> encodeFunction,
		boost::function<SharedCudaPtr<T> (SharedCudaPtrVector<char> data)> decodeFunction,
		SharedCudaPtr<T> data)
{
	auto encodedData = encodeFunction(data);
	auto decodedData = decodeFunction(encodedData);
	return data->size() == decodedData->size();
}

template<typename T>
bool EncodeDecodeUnittestHelper::TestContent2(
		boost::function<SharedCudaPtrVector<char> (SharedCudaPtr<T> data)> encodeFunction,
		boost::function<SharedCudaPtr<T> (SharedCudaPtrVector<char> data)> decodeFunction,
		SharedCudaPtr<T> data)
{
	auto encodedData = encodeFunction(data);
	auto decodedData = decodeFunction(encodedData);

//	HelperPrint::PrintTestArrays<T>(data->get(), decodedData->get(), data->size());

	return CompareDeviceArrays(data->get(), decodedData->get(), data->size());
}

#define SCALE_SPEC(X) \
	template bool EncodeDecodeUnittestHelper::TestSize<X>(								\
			boost::function<SharedCudaPtr<char> (SharedCudaPtr<X> data)> encodeFunction,\
			boost::function<SharedCudaPtr<X> (SharedCudaPtr<char> data)> decodeFunction,\
			SharedCudaPtr<X> data); 													\
	template bool EncodeDecodeUnittestHelper::TestSize2<X>(								\
			boost::function<SharedCudaPtrVector<char> (SharedCudaPtr<X> data)> encodeFunction,\
			boost::function<SharedCudaPtr<X> (SharedCudaPtrVector<char> data)> decodeFunction,\
			SharedCudaPtr<X> data); 													\
	template bool EncodeDecodeUnittestHelper::TestContent<X>(							\
			boost::function<SharedCudaPtr<char> (SharedCudaPtr<X> data)> encodeFunction,\
			boost::function<SharedCudaPtr<X> (SharedCudaPtr<char> data)> decodeFunction,\
			SharedCudaPtr<X> data);														\
	template bool EncodeDecodeUnittestHelper::TestContent2<X>(							\
			boost::function<SharedCudaPtrVector<char> (SharedCudaPtr<X> data)> encodeFunction,\
			boost::function<SharedCudaPtr<X> (SharedCudaPtrVector<char> data)> decodeFunction,\
			SharedCudaPtr<X> data);
FOR_EACH(SCALE_SPEC, float, int, long long, unsigned int)

} /* namespace ddj */
