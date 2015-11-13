/*
 *  const_encoding_unittest.cpp
 *
 *  Created on: 30-10-2015
 *      Author: Karol Dzitkowski
 */

#include "test/compression_unittest_base.hpp"
#include "const_encoding.hpp"
#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj
{

class ConstEncodingTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
    ConstEncoding_Compression_Inst,
    ConstEncodingTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(ConstEncodingTest, CompressionOfRandomInts_size)
{
    ConstEncoding encoder;
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&ConstEncoding::Encode<int>, encoder, _1),
			boost::bind(&ConstEncoding::Decode<int>, encoder, _1),
			GetIntRandomData(50,100))
    );
}

TEST_P(ConstEncodingTest, CompressionOfRandomInts_data)
{
	ConstEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&ConstEncoding::Encode<int>, encoder, _1),
			boost::bind(&ConstEncoding::Decode<int>, encoder, _1),
			GetIntRandomData(50,100))
	);
}

TEST_P(ConstEncodingTest, CompressionOfRandomFloats_size)
{
	ConstEncoding encoder;
    EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&ConstEncoding::Encode<float>, encoder, _1),
			boost::bind(&ConstEncoding::Decode<float>, encoder, _1),
			GetFloatRandomDataWithMaxPrecision(2))
    );
}

TEST_P(ConstEncodingTest, CompressionOfRandomFloats_data)
{
	ConstEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&ConstEncoding::Encode<float>, encoder, _1),
			boost::bind(&ConstEncoding::Decode<float>, encoder, _1),
			GetFloatRandomDataWithMaxPrecision(2))
	);
}

TEST_P(ConstEncodingTest, CompressionOf_Int_FromFile)
{
	ConstEncoding encoder;
    EXPECT_TRUE(
    	TestContent<time_t>(
			boost::bind(&ConstEncoding::Encode<time_t>, encoder, _1),
			boost::bind(&ConstEncoding::Decode<time_t>, encoder, _1),
			GetTsIntDataFromTestFile())
    );
}

TEST_P(ConstEncodingTest, CompressionOf_Float_FromFile)
{
	ConstEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&ConstEncoding::Encode<float>, encoder, _1),
			boost::bind(&ConstEncoding::Decode<float>, encoder, _1),
			GetTsFloatDataFromTestFile())
	);
}

TEST_P(ConstEncodingTest, GetMetadataSize_Consecutive_Int)
{
	TestGetMetadataSize<ConstEncoding, int>(GetIntConsecutiveData());
	TestGetMetadataSize<ConstEncoding, int>(GetIntRandomData(10,100));
}

TEST_P(ConstEncodingTest, GetCompressedSize_Consecutive_Int)
{
	TestGetCompressedSize<ConstEncoding, int>(GetIntConsecutiveData());
	TestGetCompressedSize<ConstEncoding, int>(GetIntRandomData(10,100));
}

} /* namespace ddj */
