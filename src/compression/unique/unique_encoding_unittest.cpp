#include "test/compression_unittest_base.hpp"
#include "unique_encoding.hpp"
#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj {

class UniqueEncodingTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
	UniqueEncoding_Compression_Inst,
	UniqueEncodingTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(UniqueEncodingTest, CompressionOfRandomInts_size)
{
	UniqueEncoding encoder;
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&UniqueEncoding::Encode<int>, encoder, _1),
			boost::bind(&UniqueEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
    );
}

TEST_P(UniqueEncodingTest, CompressionOfRandomInts_data)
{
	UniqueEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&UniqueEncoding::Encode<int>, encoder, _1),
			boost::bind(&UniqueEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
	);
}

TEST_P(UniqueEncodingTest, CompressionOfRandomFloats_size)
{
	UniqueEncoding encoder;
    EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&UniqueEncoding::Encode<float>, encoder, _1),
			boost::bind(&UniqueEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
    );
}

TEST_P(UniqueEncodingTest, CompressionOfRandomFloats_data)
{
	UniqueEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&UniqueEncoding::Encode<float>, encoder, _1),
			boost::bind(&UniqueEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
	);
}

TEST_P(UniqueEncodingTest, GetMetadataSize_Consecutive_Int)
{
	TestGetMetadataSize<UniqueEncoding, int>(GetIntConsecutiveData());
}

TEST_P(UniqueEncodingTest, GetCompressedSize_Consecutive_Int)
{
	TestGetCompressedSize<UniqueEncoding, int>(GetIntConsecutiveData());
}

} /* namespace ddj */
