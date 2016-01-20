#include "test/compression_unittest_base.hpp"
#include "delta_encoding.hpp"
#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj {

class DeltaCompressionTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
	DeltaEncoding_Compression_Inst,
	DeltaCompressionTest,
    ::testing::Values(100, 100000));

TEST_P(DeltaCompressionTest, CompressionOfRandomInts_size)
{
	DeltaEncoding encoder;
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&DeltaEncoding::Encode<int>, encoder, _1),
			boost::bind(&DeltaEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
    );
}

TEST_P(DeltaCompressionTest, CompressionOfRandomInts_data)
{
	DeltaEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&DeltaEncoding::Encode<int>, encoder, _1),
			boost::bind(&DeltaEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
	);
}

TEST_P(DeltaCompressionTest, CompressionOfRandomFloats_size)
{
	DeltaEncoding encoder;
    EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&DeltaEncoding::Encode<float>, encoder, _1),
			boost::bind(&DeltaEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
    );
}

TEST_P(DeltaCompressionTest, CompressionOfRandomFloats_data)
{
	DeltaEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&DeltaEncoding::Encode<float>, encoder, _1),
			boost::bind(&DeltaEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
	);
}

TEST_P(DeltaCompressionTest, CompressionOf_PatternA_Short)
{
	DeltaEncoding encoder;
	EXPECT_TRUE(
		TestContent<short>(
			boost::bind(&DeltaEncoding::Encode<short>, encoder, _1),
			boost::bind(&DeltaEncoding::Decode<short>, encoder, _1),
			GetFakeDataWithPatternA<short>(0, GetSize()/3, 1, 0, 1e3))
	);
}

TEST_P(DeltaCompressionTest, CompressionOf_PatternA_Char)
{
	DeltaEncoding encoder;
	EXPECT_TRUE(
		TestContent<char>(
			boost::bind(&DeltaEncoding::Encode<char>, encoder, _1),
			boost::bind(&DeltaEncoding::Decode<char>, encoder, _1),
			GetFakeDataWithPatternA<char>(0, GetSize()/3, 1, 0, 1e2))
	);
}

TEST_P(DeltaCompressionTest, GetMetadataSize_Consecutive_Int)
{
	TestGetMetadataSize<DeltaEncoding, int>(GetIntConsecutiveData());
}

TEST_P(DeltaCompressionTest, GetCompressedSize_Consecutive_Int)
{
	TestGetCompressedSize<DeltaEncoding, int>(GetIntConsecutiveData());
}

TEST_P(DeltaCompressionTest, GetMetadataSize_Random_Int)
{
	TestGetMetadataSize<DeltaEncoding, int>(GetIntRandomData(10,100));
}

TEST_P(DeltaCompressionTest, GetCompressedSize_Random_Int)
{
	TestGetCompressedSize<DeltaEncoding, int>(GetIntRandomData(10,100));
}

TEST_P(DeltaCompressionTest, GetMetadataSize_RandomFloatsWithMaxPrecision2)
{
	TestGetMetadataSize<DeltaEncoding, float>(GetFloatRandomDataWithMaxPrecision(2));
}

TEST_P(DeltaCompressionTest, GetCompressedSize_RandomFloatsWithMaxPrecision2)
{
	TestGetCompressedSize<DeltaEncoding, float>(GetFloatRandomDataWithMaxPrecision(2));
}


} /* namespace ddj */
