/*
 * float_encoding_unitttest.cpp
 *
 *  Created on: 30 paź 2015
 *      Author: ghash
 */

#include "test/compression_unittest_base.hpp"
#include "float_encoding.hpp"
#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj
{

class FloatEncodingTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
    FloatEncoding_Compression_Inst,
    FloatEncodingTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(FloatEncodingTest, CompressionOfRandomFloats_WithMaxPrecision_3_size)
{
    FloatEncoding encoder;
    EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&FloatEncoding::Encode<float>, encoder, _1),
			boost::bind(&FloatEncoding::Decode<float>, encoder, _1),
			GetFloatRandomDataWithMaxPrecision(3))
    );
}

TEST_P(FloatEncodingTest, CompressionOfRandomFloats_WithMaxPrecision_3_data)
{
	FloatEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&FloatEncoding::Encode<float>, encoder, _1),
			boost::bind(&FloatEncoding::Decode<float>, encoder, _1),
			GetFloatRandomDataWithMaxPrecision(3))
	);
}

TEST_P(FloatEncodingTest, CompressionOf_Float_FromFile)
{
	FloatEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&FloatEncoding::Encode<float>, encoder, _1),
			boost::bind(&FloatEncoding::Decode<float>, encoder, _1),
			GetTsFloatDataFromTestFile())
	);
}

TEST_P(FloatEncodingTest, GetMetadataSize_Consecutive_Int)
{
	TestGetMetadataSize<FloatEncoding, int>(GetIntConsecutiveData());
}

TEST_P(FloatEncodingTest, GetCompressedSize_Consecutive_Int)
{
	TestGetCompressedSize<FloatEncoding, int>(GetIntConsecutiveData());
}

} /* namespace ddj */