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

class ConstCompressionTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
    ConstEncoding_Compression_Inst,
    ConstCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(ConstCompressionTest, CompressionOfRandomInts_size)
{
    ConstEncoding encoder;
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&ConstEncoding::Encode<int>, encoder, _1),
			boost::bind(&ConstEncoding::Decode<int>, encoder, _1),
			GetIntRandomData(50,100))
    );
}

TEST_P(ConstCompressionTest, CompressionOfRandomInts_data)
{
	ConstEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&ConstEncoding::Encode<int>, encoder, _1),
			boost::bind(&ConstEncoding::Decode<int>, encoder, _1),
			GetIntRandomData(50,100))
	);
}

TEST_P(ConstCompressionTest, CompressionOfRandomFloats_size)
{
	ConstEncoding encoder;
    EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&ConstEncoding::Encode<float>, encoder, _1),
			boost::bind(&ConstEncoding::Decode<float>, encoder, _1),
			GetFloatRandomDataWithMaxPrecision(2))
    );
}

TEST_P(ConstCompressionTest, CompressionOfRandomFloats_data)
{
	ConstEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&ConstEncoding::Encode<float>, encoder, _1),
			boost::bind(&ConstEncoding::Decode<float>, encoder, _1),
			GetFloatRandomDataWithMaxPrecision(2))
	);
}

} /* namespace ddj */
