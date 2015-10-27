#include "test/compression_unittest_base.hpp"
#include "dict_encoding.hpp"
#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj {

class DictCompressionTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
	DictEncoding_Compression_Inst,
	DictCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(DictCompressionTest, CompressionOfRandomInts_size)
{
	DictEncoding encoder;
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&DictEncoding::Encode<int>, encoder, _1),
			boost::bind(&DictEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
    );
}

TEST_P(DictCompressionTest, CompressionOfRandomInts_data)
{
	DictEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&DictEncoding::Encode<int>, encoder, _1),
			boost::bind(&DictEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
	);
}

TEST_P(DictCompressionTest, CompressionOfRandomFloats_size)
{
	DictEncoding encoder;
    EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&DictEncoding::Encode<float>, encoder, _1),
			boost::bind(&DictEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
    );
}

TEST_P(DictCompressionTest, CompressionOfRandomFloats_data)
{
	DictEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&DictEncoding::Encode<float>, encoder, _1),
			boost::bind(&DictEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
	);
}


} /* namespace ddj */
