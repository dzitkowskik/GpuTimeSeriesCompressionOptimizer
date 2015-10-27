#include "test/compression_unittest_base.hpp"
#include "unique_encoding.hpp"
#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj {

class UniqueCompressionTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
	UniqueEncoding_Compression_Inst,
	UniqueCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(UniqueCompressionTest, CompressionOfRandomInts_size)
{
	UniqueEncoding encoder;
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&UniqueEncoding::Encode<int>, encoder, _1),
			boost::bind(&UniqueEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
    );
}

TEST_P(UniqueCompressionTest, CompressionOfRandomInts_data)
{
	UniqueEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&UniqueEncoding::Encode<int>, encoder, _1),
			boost::bind(&UniqueEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
	);
}

TEST_P(UniqueCompressionTest, CompressionOfRandomFloats_size)
{
	UniqueEncoding encoder;
    EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&UniqueEncoding::Encode<float>, encoder, _1),
			boost::bind(&UniqueEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
    );
}

TEST_P(UniqueCompressionTest, CompressionOfRandomFloats_data)
{
	UniqueEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&UniqueEncoding::Encode<float>, encoder, _1),
			boost::bind(&UniqueEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
	);
}


} /* namespace ddj */
