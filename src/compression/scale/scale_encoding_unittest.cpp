#include "test/compression_unittest_base.hpp"
#include "scale_encoding.hpp"
#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj {

class ScaleCompressionTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
    ScaleEncoding_Compression_Inst,
    ScaleCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(ScaleCompressionTest, CompressionOfRandomInts_size)
{
    ScaleEncoding encoder;
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&ScaleEncoding::Encode<int>, encoder, _1),
			boost::bind(&ScaleEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
    );
}

TEST_P(ScaleCompressionTest, CompressionOfRandomInts_data)
{
	ScaleEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&ScaleEncoding::Encode<int>, encoder, _1),
			boost::bind(&ScaleEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
	);
}

TEST_P(ScaleCompressionTest, CompressionOfRandomFloats_size)
{
    ScaleEncoding encoder;
    EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&ScaleEncoding::Encode<float>, encoder, _1),
			boost::bind(&ScaleEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
    );
}

TEST_P(ScaleCompressionTest, CompressionOfRandomFloats_data)
{
	ScaleEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&ScaleEncoding::Encode<float>, encoder, _1),
			boost::bind(&ScaleEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
	);
}


} /* namespace ddj */
