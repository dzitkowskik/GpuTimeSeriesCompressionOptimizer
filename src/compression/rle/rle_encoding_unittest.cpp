#include "test/compression_unittest_base.hpp"
#include "rle_encoding.hpp"
#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj {

class RleCompressionTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
	RleEncoding_Compression_Inst,
	RleCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(RleCompressionTest, CompressionOfRandomInts_size)
{
	RleEncoding encoder;
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&RleEncoding::Encode<int>, encoder, _1),
			boost::bind(&RleEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
    );
}

TEST_P(RleCompressionTest, CompressionOfRandomInts_data)
{
	RleEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&RleEncoding::Encode<int>, encoder, _1),
			boost::bind(&RleEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
	);
}

TEST_P(RleCompressionTest, CompressionOfRandomFloats_size)
{
	RleEncoding encoder;
    EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&RleEncoding::Encode<float>, encoder, _1),
			boost::bind(&RleEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
    );
}

TEST_P(RleCompressionTest, CompressionOfRandomFloats_data)
{
	RleEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&RleEncoding::Encode<float>, encoder, _1),
			boost::bind(&RleEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
	);
}


} /* namespace ddj */
