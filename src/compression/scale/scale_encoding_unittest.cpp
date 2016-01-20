#include "test/compression_unittest_base.hpp"
#include "scale_encoding.hpp"
#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj {

class ScaleCompressionTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
    ScaleEncoding_Compression_Inst,
    ScaleCompressionTest,
    ::testing::Values(100, 100000));

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

TEST_P(ScaleCompressionTest, CompressionOf_PatternA_Short)
{
	ScaleEncoding encoder;
	EXPECT_TRUE(
		TestContent<short>(
			boost::bind(&ScaleEncoding::Encode<short>, encoder, _1),
			boost::bind(&ScaleEncoding::Decode<short>, encoder, _1),
			GetFakeDataWithPatternA<short>(0, GetSize()/3, 1, 0, 1e3))
	);
}

TEST_P(ScaleCompressionTest, CompressionOf_PatternA_Char)
{
	ScaleEncoding encoder;
	EXPECT_TRUE(
		TestContent<char>(
			boost::bind(&ScaleEncoding::Encode<char>, encoder, _1),
			boost::bind(&ScaleEncoding::Decode<char>, encoder, _1),
			GetFakeDataWithPatternA<char>(0, GetSize()/3, 1, 0, 1e2))
	);
}

TEST_P(ScaleCompressionTest, GetMetadataSize_Consecutive_Int)
{
	TestGetMetadataSize<ScaleEncoding, int>(GetIntConsecutiveData());
	TestGetMetadataSize<ScaleEncoding, float>(GetFloatRandomDataWithMaxPrecision(2));
}

TEST_P(ScaleCompressionTest, GetCompressedSize_Consecutive_Int)
{
	TestGetCompressedSize<ScaleEncoding, int>(GetIntConsecutiveData());
	TestGetCompressedSize<ScaleEncoding, float>(GetFloatRandomDataWithMaxPrecision(2));
}

} /* namespace ddj */
