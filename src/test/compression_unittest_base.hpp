/*
 *  compression_unittest_base.hpp
 *
 *  Created on: 22-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_COMPRESSION_UNITTEST_BASE_HPP_
#define DDJ_COMPRESSION_UNITTEST_BASE_HPP_

#include "unittest_base.hpp"
#include "compression/encoding.hpp"

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

	template<class EncodingClass, typename T>
	static bool TestGetMetadataSize(SharedCudaPtr<T> data)
	{
		boost::shared_ptr<Encoding> encoder = boost::make_shared<EncodingClass>();
		auto testData = MoveSharedCudaPtr<T, char>(data);
		auto encoded = encoder->Encode(testData->copy(), GetDataType<T>());
		auto expected = encoded[0]->size();
		auto actual = encoder->GetMetadataSize(testData, GetDataType<T>());
		EXPECT_EQ(expected, actual);
	}

protected:
	void SetUp()
	{
		_size = GetParam();
	}
};

} /* namespace ddj */
#endif /* DDJ_COMPRESSION_UNITTEST_HPP_ */
