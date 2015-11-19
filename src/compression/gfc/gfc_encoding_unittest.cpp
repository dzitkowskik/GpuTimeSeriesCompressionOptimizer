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
    ::testing::Values(10, 1000, 10000));

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

} /* namespace ddj */
