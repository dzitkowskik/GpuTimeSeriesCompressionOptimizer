#include "test/compression_unittest_base.hpp"
#include "afl_encoding.hpp"
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

} /* namespace ddj */
