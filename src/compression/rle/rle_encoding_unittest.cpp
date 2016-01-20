#include "test/compression_unittest_base.hpp"
#include "rle_encoding.hpp"
#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj {

class RleCompressionTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
	RleEncoding_Compression_Inst,
	RleCompressionTest,
    ::testing::Values(100, 100000));

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

TEST_P(RleCompressionTest, CompressionOf_PatternA_Short)
{
	RleEncoding encoder;
	EXPECT_TRUE(
		TestContent<short>(
			boost::bind(&RleEncoding::Encode<short>, encoder, _1),
			boost::bind(&RleEncoding::Decode<short>, encoder, _1),
			GetFakeDataWithPatternA<short>(0, GetSize()/3, 1, 0, 1e3))
	);
}

TEST_P(RleCompressionTest, CompressionOf_PatternA_Char)
{
	RleEncoding encoder;
	EXPECT_TRUE(
		TestContent<char>(
			boost::bind(&RleEncoding::Encode<char>, encoder, _1),
			boost::bind(&RleEncoding::Decode<char>, encoder, _1),
			GetFakeDataWithPatternA<char>(0, GetSize()/3, 1, 0, 1e2))
	);
}

TEST_P(RleCompressionTest, GetMetadataSize)
{
	TestGetMetadataSize<RleEncoding, int>(GetIntConsecutiveData());
	TestGetMetadataSize<RleEncoding, int>(GetIntRandomData(10,100));
	TestGetMetadataSize<RleEncoding, float>(GetFloatRandomDataWithMaxPrecision(2));
}

TEST_P(RleCompressionTest, GetCompressedSize)
{
	TestGetCompressedSize<RleEncoding, int>(GetIntConsecutiveData());
	TestGetCompressedSize<RleEncoding, int>(GetIntRandomData(10,100));
	TestGetCompressedSize<RleEncoding, float>(GetFloatRandomDataWithMaxPrecision(2));
}


} /* namespace ddj */
