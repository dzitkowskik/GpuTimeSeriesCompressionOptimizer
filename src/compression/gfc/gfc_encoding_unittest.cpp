/*
 * gfc_encoding_unittest.cpp
 *
 *  Created on: Nov 17, 2015
 *      Author: Karol Dzitkowski
 */

#include "compression/gfc/gfc_encoding.hpp"
#include "test/compression_unittest_base.hpp"
#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj
{

class GfcCompressionTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
	GfcEncoding_Compression_Inst,
	GfcCompressionTest,
    ::testing::Values(100, 100000));

TEST_P(GfcCompressionTest, CompressionOfRandomDoubles_size)
{
	GfcEncoding encoder;
    EXPECT_TRUE(
		TestSize<double>(
			boost::bind(&GfcEncoding::Encode<double>, encoder, _1),
			boost::bind(&GfcEncoding::Decode<double>, encoder, _1),
			GetDoubleRandomData())
    );
}

TEST_P(GfcCompressionTest, CompressionOfRandomDoubles_data)
{
	GfcEncoding encoder;
	EXPECT_TRUE(
		TestContent<double>(
			boost::bind(&GfcEncoding::Encode<double>, encoder, _1),
			boost::bind(&GfcEncoding::Decode<double>, encoder, _1),
			GetDoubleRandomData())
	);
}

TEST_P(GfcCompressionTest, CompressionOfRandomFloats_size)
{
	GfcEncoding encoder;
    EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&GfcEncoding::Encode<float>, encoder, _1),
			boost::bind(&GfcEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
    );
}

TEST_P(GfcCompressionTest, CompressionOfRandomFloats_data)
{
	GfcEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&GfcEncoding::Encode<float>, encoder, _1),
			boost::bind(&GfcEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
	);
}

TEST_P(GfcCompressionTest, CompressionOfRealDataFloats_data)
{
	GfcEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&GfcEncoding::Encode<float>, encoder, _1),
			boost::bind(&GfcEncoding::Decode<float>, encoder, _1),
			GetTsFloatDataFromTestFile())
	);
}

TEST_P(GfcCompressionTest, CompressionOfFakeFloats_PatternA_data)
{
	GfcEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&GfcEncoding::Encode<float>, encoder, _1),
			boost::bind(&GfcEncoding::Decode<float>, encoder, _1),
			GetFakeDataWithPatternA<float>(0))
	);
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&GfcEncoding::Encode<float>, encoder, _1),
			boost::bind(&GfcEncoding::Decode<float>, encoder, _1),
			GetFakeDataWithPatternA<float>(1, 1e3, 0.1, -3.0, 3e6))
	);	
}

} /* namespace ddj */
