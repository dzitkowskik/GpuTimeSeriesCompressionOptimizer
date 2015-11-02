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

TEST_P(AflCompressionTest, Afl_GetCompressedSize_int)
{
	AflEncoding encoder;
	auto randomIntData = GetIntConsecutiveData();
	SharedCudaPtr<char> charData = boost::reinterpret_pointer_cast<CudaPtr<char>>(randomIntData);
	size_t actual = encoder.GetCompressedSize(charData, DataType::d_int);
	auto minBit = CudaArrayStatistics().MinBitCnt<int>(randomIntData);
	size_t expected = ceil((double)(randomIntData->size() * minBit) / 32.0) * 4;
	EXPECT_EQ(expected, actual);
}

} /* namespace ddj */
