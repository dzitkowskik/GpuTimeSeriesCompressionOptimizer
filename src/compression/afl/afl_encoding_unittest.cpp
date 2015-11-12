#include "test/compression_unittest_base.hpp"
#include "afl_encoding.hpp"
#include "util/statistics/cuda_array_statistics.hpp"

#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj {

class AflCompressionTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
	AflEncoding_Compression_Inst,
	AflCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(AflCompressionTest, Afl_Encode_Decode_RandomInts_size)
{
	AflEncoding encoder;
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&AflEncoding::Encode<int>, encoder, _1),
			boost::bind(&AflEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
    );
}

TEST_P(AflCompressionTest, Afl_Encode_Decode_RandomInts_data)
{
	AflEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&AflEncoding::Encode<int>, encoder, _1),
			boost::bind(&AflEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
	);
}

TEST_P(AflCompressionTest, Afl_Encode_Decode_RandomFloats_size)
{
	AflEncoding encoder;
    EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&AflEncoding::Encode<float>, encoder, _1),
			boost::bind(&AflEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
    );
}

TEST_P(AflCompressionTest, Afl_Encode_Decode_RandomFloats_data)
{
	AflEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&AflEncoding::Encode<float>, encoder, _1),
			boost::bind(&AflEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
	);
}

TEST_P(AflCompressionTest, Afl_Encode_Decode_RandomFloatsWithMaxPrecision2_data)
{
	AflEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&AflEncoding::Encode<float>, encoder, _1),
			boost::bind(&AflEncoding::Decode<float>, encoder, _1),
			GetFloatRandomDataWithMaxPrecision(2))
	);
}

TEST_P(AflCompressionTest, Afl_Encode_Decode_ConsecutiveInts_size)
{
	AflEncoding encoder;
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&AflEncoding::Encode<int>, encoder, _1),
			boost::bind(&AflEncoding::Decode<int>, encoder, _1),
			GetIntConsecutiveData())
    );
}

TEST_P(AflCompressionTest, Afl_Encode_Decode_ConsecutiveInts_data)
{
	AflEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&AflEncoding::Encode<int>, encoder, _1),
			boost::bind(&AflEncoding::Decode<int>, encoder, _1),
			GetIntConsecutiveData())
	);
}

TEST_P(AflCompressionTest, GetMetadataSize_Consecutive_Int)
{
	TestGetMetadataSize<AflEncoding, int>(GetIntConsecutiveData());
}

TEST_P(AflCompressionTest, GetCompressedSize_Consecutive_Int)
{
	TestGetCompressedSize<AflEncoding, int>(GetIntConsecutiveData());
}

TEST_P(AflCompressionTest, GetMetadataSize_RandomFloatsWithMaxPrecision2)
{
	TestGetMetadataSize<AflEncoding, float>(GetFloatRandomDataWithMaxPrecision(2));
}

TEST_P(AflCompressionTest, GetCompressedSize_RandomFloatsWithMaxPrecision2)
{
	TestGetCompressedSize<AflEncoding, float>(GetFloatRandomDataWithMaxPrecision(2));
}

} /* namespace ddj */
