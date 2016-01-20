#include "test/compression_unittest_base.hpp"
#include "dict_encoding.hpp"
#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj {

class DictEncodingTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
	DictEncoding_Compression_Inst,
	DictEncodingTest,
    ::testing::Values(100, 100000));

TEST_P(DictEncodingTest, CompressionOfRandomInts_size)
{
	DictEncoding encoder;
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&DictEncoding::Encode<int>, encoder, _1),
			boost::bind(&DictEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
    );
}

TEST_P(DictEncodingTest, CompressionOfRandomInts_data)
{
	DictEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&DictEncoding::Encode<int>, encoder, _1),
			boost::bind(&DictEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
	);
}

TEST_P(DictEncodingTest, DISABLED_CompressionOfRandomInts_bigData)
{
	DictEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&DictEncoding::Encode<int>, encoder, _1),
			boost::bind(&DictEncoding::Decode<int>, encoder, _1),
			CudaArrayGenerator().GenerateRandomIntDeviceArray(1<<20, 10, 1000))
	);
}

TEST_P(DictEncodingTest, CompressionOfRandomFloats_size)
{
	DictEncoding encoder;
    EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&DictEncoding::Encode<float>, encoder, _1),
			boost::bind(&DictEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
    );
}

TEST_P(DictEncodingTest, CompressionOfRandomFloats_data)
{
	DictEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&DictEncoding::Encode<float>, encoder, _1),
			boost::bind(&DictEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
	);
}

TEST_P(DictEncodingTest, CompressionOf_Long_FromFile_data)
{
	DictEncoding encoder;
    EXPECT_TRUE(
		TestContent<time_t>(
			boost::bind(&DictEncoding::Encode<time_t>, encoder, _1),
			boost::bind(&DictEncoding::Decode<time_t>, encoder, _1),
			GetTsIntDataFromTestFile())
    );
}

TEST_P(DictEncodingTest, CompressionOf_Float_FromFile_data)
{
	DictEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&DictEncoding::Encode<float>, encoder, _1),
			boost::bind(&DictEncoding::Decode<float>, encoder, _1),
			GetTsFloatDataFromTestFile())
	);
}

TEST_P(DictEncodingTest, CompressionOf_RandomFloatsWithMaxPrecision2_data)
{
	DictEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&DictEncoding::Encode<float>, encoder, _1),
			boost::bind(&DictEncoding::Decode<float>, encoder, _1),
			GetFloatRandomDataWithMaxPrecision(2))
	);
}

TEST_P(DictEncodingTest, CompressionOf_PatternA_Short)
{
	DictEncoding encoder;
	EXPECT_TRUE(
		TestContent<short>(
			boost::bind(&DictEncoding::Encode<short>, encoder, _1),
			boost::bind(&DictEncoding::Decode<short>, encoder, _1),
			GetFakeDataWithPatternA<short>(0, GetSize()/3, 1, 0, 1e3))
	);
}

TEST_P(DictEncodingTest, CompressionOf_PatternA_Char)
{
	DictEncoding encoder;
	EXPECT_TRUE(
		TestContent<char>(
			boost::bind(&DictEncoding::Encode<char>, encoder, _1),
			boost::bind(&DictEncoding::Decode<char>, encoder, _1),
			GetFakeDataWithPatternA<char>(0, GetSize()/3, 1, 0, 1e2))
	);
}

TEST_P(DictEncodingTest, GetMetadataSize_Consecutive_Int)
{
	TestGetMetadataSize<DictEncoding, int>(GetIntConsecutiveData());
	TestGetMetadataSize<DictEncoding, int>(GetIntRandomData(10,100));
	TestGetMetadataSize<DictEncoding, float>(GetFloatRandomDataWithMaxPrecision(2));

}

TEST_P(DictEncodingTest, GetCompressedSize_Consecutive_Int)
{
	TestGetCompressedSize<DictEncoding, int>(GetIntConsecutiveData());
	TestGetCompressedSize<DictEncoding, int>(GetIntRandomData(10,100));
	TestGetCompressedSize<DictEncoding, float>(GetFloatRandomDataWithMaxPrecision(2));
}

} /* namespace ddj */
