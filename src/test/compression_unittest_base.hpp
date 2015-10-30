/*
 *  compression_unittest_base.hpp
 *
 *  Created on: 22-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_COMPRESSION_UNITTEST_BASE_HPP_
#define DDJ_COMPRESSION_UNITTEST_BASE_HPP_

#include "unittest_base.hpp"

namespace ddj
{

class CompressionUnittestBase : public UnittestBase, public ::testing::WithParamInterface<int>
{
public:
	template<typename T>
	static bool TestSize(
			boost::function<SharedCudaPtrVector<char> (SharedCudaPtr<T> data)> encodeFunction,
			boost::function<SharedCudaPtr<T> (SharedCudaPtrVector<char> data)> decodeFunction,
			SharedCudaPtr<T> data);

	template<typename T>
	static bool TestContent(
			boost::function<SharedCudaPtrVector<char> (SharedCudaPtr<T>)> encodeFunction,
			boost::function<SharedCudaPtr<T> (SharedCudaPtrVector<char>)> decodeFunction,
			SharedCudaPtr<T> data);
protected:
	void SetUp()
	{
		_size = GetParam();
	}
};

} /* namespace ddj */
#endif /* DDJ_COMPRESSION_UNITTEST_HPP_ */
