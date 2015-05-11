/*
 * encode_decode_unittest_helper.hpp
 *
 *  Created on: 07-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef ENCODE_DECODE_UNITTEST_HELPER_H_
#define ENCODE_DECODE_UNITTEST_HELPER_H_

#include <boost/function.hpp>
#include <boost/bind.hpp>
#include "core/cuda_ptr.hpp"

namespace ddj
{

class EncodeDecodeUnittestHelper
{
public:
	template<typename T>
	static bool TestSize(
			boost::function<SharedCudaPtr<char> (SharedCudaPtr<T> data)> encodeFunction,
			boost::function<SharedCudaPtr<T> (SharedCudaPtr<char> data)> decodeFunction,
			SharedCudaPtr<T> data);

	template<typename T>
	static bool TestContent(
			boost::function<SharedCudaPtr<char> (SharedCudaPtr<T>)> encodeFunction,
			boost::function<SharedCudaPtr<T> (SharedCudaPtr<char>)> decodeFunction,
			SharedCudaPtr<T> data);
};

} /* namespace ddj */
#endif /* ENCODE_DECODE_UNITTEST_HELPER_H_ */
