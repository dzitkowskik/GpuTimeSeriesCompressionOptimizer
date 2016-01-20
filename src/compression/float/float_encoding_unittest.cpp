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
    ::testing::Values(100, 100000));

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

TEST_P(FloatEncodingTest, CompressionOf_PatternA_Short)
{
	FloatEncoding encoder;
	EXPECT_TRUE(
		TestContent<short>(
			boost::bind(&FloatEncoding::Encode<short>, encoder, _1),
			boost::bind(&FloatEncoding::Decode<short>, encoder, _1),
			GetFakeDataWithPatternA<short>(0, GetSize()/3, 1, 0, 1e3))
	);
}

TEST_P(FloatEncodingTest, CompressionOf_PatternA_Char)
{
	FloatEncoding encoder;
	EXPECT_TRUE(
		TestContent<char>(
			boost::bind(&FloatEncoding::Encode<char>, encoder, _1),
			boost::bind(&FloatEncoding::Decode<char>, encoder, _1),
			GetFakeDataWithPatternA<char>(0, GetSize()/3, 1, 0, 1e2))
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
