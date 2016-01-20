#include "test/compression_unittest_base.hpp"
#include "unique_encoding.hpp"
#include "helpers/helper_print.hpp"
#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj {

class UniqueEncodingTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
	UniqueEncoding_Compression_Inst,
	UniqueEncodingTest,
    ::testing::Values(100, 100000));

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

TEST_P(UniqueEncodingTest, CompressionOfRandomBinaryValues_data)
{
	auto binaryValues = GetIntRandomData(0,2);
	UniqueEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&UniqueEncoding::Encode<int>, encoder, _1),
			boost::bind(&UniqueEncoding::Decode<int>, encoder, _1),
			binaryValues)
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

TEST_P(UniqueEncodingTest, CompressionOf_Long_FromFile)
{
	UniqueEncoding encoder;
    EXPECT_TRUE(
		TestContent<time_t>(
			boost::bind(&UniqueEncoding::Encode<time_t>, encoder, _1),
			boost::bind(&UniqueEncoding::Decode<time_t>, encoder, _1),
			GetTsIntDataFromTestFile())
    );
}

TEST_P(UniqueEncodingTest, CompressionOf_PatternA_Short)
{
	UniqueEncoding encoder;
	EXPECT_TRUE(
		TestContent<short>(
			boost::bind(&UniqueEncoding::Encode<short>, encoder, _1),
			boost::bind(&UniqueEncoding::Decode<short>, encoder, _1),
			GetFakeDataWithPatternA<short>(0, GetSize()/3, 1, 0, 1e3))
	);
}

TEST_P(UniqueEncodingTest, CompressionOf_PatternA_Char)
{
	UniqueEncoding encoder;
	EXPECT_TRUE(
		TestContent<char>(
			boost::bind(&UniqueEncoding::Encode<char>, encoder, _1),
			boost::bind(&UniqueEncoding::Decode<char>, encoder, _1),
			GetFakeDataWithPatternA<char>(0, GetSize()/3, 1, 0, 1e2))
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
