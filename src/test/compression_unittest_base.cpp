/*
 * encode_decode_unittest_helper.cpp
 *
 *  Created on: 07-05-2015
 *      Author: Karol Dzitkowski
 */

#include "compression_unittest_base.hpp"
#include "helpers/helper_comparison.cuh"
#include "core/macros.h"

#include <boost/function.hpp>

namespace ddj
{

template<typename T>
bool CompressionUnittestBase::TestSize(
		boost::function<SharedCudaPtrVector<char> (SharedCudaPtr<T> data)> encodeFunction,
		boost::function<SharedCudaPtr<T> (SharedCudaPtrVector<char> data)> decodeFunction,
		SharedCudaPtr<T> data)
{
	auto encodedData = encodeFunction(data);
	auto decodedData = decodeFunction(encodedData);
	return data->size() == decodedData->size();
}

template<typename T>
bool CompressionUnittestBase::TestContent(
		boost::function<SharedCudaPtrVector<char> (SharedCudaPtr<T> data)> encodeFunction,
		boost::function<SharedCudaPtr<T> (SharedCudaPtrVector<char> data)> decodeFunction,
		SharedCudaPtr<T> data)
{
	auto encodedData = encodeFunction(data);
	auto decodedData = decodeFunction(encodedData);
	return CompareDeviceArrays(data->get(), decodedData->get(), data->size());
}

#define COMPRESSION_UNITTEST_BASE_SPEC(X) \
	template bool CompressionUnittestBase::TestSize<X>(							\
			boost::function<SharedCudaPtrVector<char> (SharedCudaPtr<X> data)>,		\
			boost::function<SharedCudaPtr<X> (SharedCudaPtrVector<char> data)>,		\
			SharedCudaPtr<X> data); 												\
	template bool CompressionUnittestBase::TestContent<X>(						\
			boost::function<SharedCudaPtrVector<char> (SharedCudaPtr<X> data)>, 	\
			boost::function<SharedCudaPtr<X> (SharedCudaPtrVector<char> data)>, 	\
			SharedCudaPtr<X> data);
FOR_EACH(COMPRESSION_UNITTEST_BASE_SPEC, float, int, long long, unsigned int)

} /* namespace ddj */
