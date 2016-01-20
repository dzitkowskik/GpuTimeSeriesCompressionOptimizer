#include "test/compression_unittest_base.hpp"
#include "patch_encoding.hpp"
#include "util/stencil/stencil_operators.hpp"
#include "compression/patch/patch_encoding_factory.hpp"
#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj {

class PatchCompressionTest : public CompressionUnittestBase {};
class PatchTest : public UnittestBase {};

INSTANTIATE_TEST_CASE_P(
	PatchEncoding_Compression_Inst,
	PatchCompressionTest,
    ::testing::Values(100, 100000));

TEST_P(PatchCompressionTest, OutsidePatch_CompressionOfRandomInts_size)
{
	OutsidePatchEncoding encoder(501, 5000);
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&OutsidePatchEncoding::Encode<int>, encoder, _1),
			boost::bind(&OutsidePatchEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
    );
    LowerPatchEncoding encoder2(501, 5000);
	EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&LowerPatchEncoding::Encode<int>, encoder2, _1),
			boost::bind(&LowerPatchEncoding::Decode<int>, encoder2, _1),
			GetIntRandomData())
	);
}

TEST_P(PatchCompressionTest, OutsidePatch_CompressionOfRandomInts_data)
{
	OutsidePatchEncoding encoder(501, 5000);
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&OutsidePatchEncoding::Encode<int>, encoder, _1),
			boost::bind(&OutsidePatchEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
	);
	LowerPatchEncoding encoder2(501, 5000);
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&LowerPatchEncoding::Encode<int>, encoder2, _1),
			boost::bind(&LowerPatchEncoding::Decode<int>, encoder2, _1),
			GetIntRandomData())
	);
}

TEST_P(PatchCompressionTest, DISABLED_OutsidePatch_CompressionOfRandomInts_bigData)
{
	OutsidePatchEncoding encoder(501, 5000);
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&OutsidePatchEncoding::Encode<int>, encoder, _1),
			boost::bind(&OutsidePatchEncoding::Decode<int>, encoder, _1),
			CudaArrayGenerator().GenerateRandomIntDeviceArray(1<<20, 10, 1000))
	);
	LowerPatchEncoding encoder2(501, 5000);
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&LowerPatchEncoding::Encode<int>, encoder2, _1),
			boost::bind(&LowerPatchEncoding::Decode<int>, encoder2, _1),
			CudaArrayGenerator().GenerateRandomIntDeviceArray(1<<20, 10, 1000))
	);
}

TEST_P(PatchCompressionTest, OutsidePatch_CompressionOfRandomFloats_size)
{
	OutsidePatchEncoding encoder(501, 5000);
    EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&OutsidePatchEncoding::Encode<float>, encoder, _1),
			boost::bind(&OutsidePatchEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
    );
    LowerPatchEncoding encoder2(501, 5000);
	EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&LowerPatchEncoding::Encode<float>, encoder2, _1),
			boost::bind(&LowerPatchEncoding::Decode<float>, encoder2, _1),
			GetFloatRandomData())
	);
}

TEST_P(PatchCompressionTest, OutsidePatch_CompressionOfRandomFloats_data)
{
	OutsidePatchEncoding encoder(501, 5000);
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&OutsidePatchEncoding::Encode<float>, encoder, _1),
			boost::bind(&OutsidePatchEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
	);
	LowerPatchEncoding encoder2(501, 5000);
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&LowerPatchEncoding::Encode<float>, encoder2, _1),
			boost::bind(&LowerPatchEncoding::Decode<float>, encoder2, _1),
			GetFloatRandomData())
	);
}

TEST_P(PatchCompressionTest, CompressionOf_PatternA_Short)
{
	OutsidePatchEncoding encoder;
	EXPECT_TRUE(
		TestContent<short>(
			boost::bind(&OutsidePatchEncoding::Encode<short>, encoder, _1),
			boost::bind(&OutsidePatchEncoding::Decode<short>, encoder, _1),
			GetFakeDataWithPatternA<short>(0, GetSize()/3, 1, 0, 1e3))
	);
	LowerPatchEncoding encoder2;
	EXPECT_TRUE(
		TestContent<short>(
			boost::bind(&LowerPatchEncoding::Encode<short>, encoder2, _1),
			boost::bind(&LowerPatchEncoding::Decode<short>, encoder2, _1),
			GetFakeDataWithPatternA<short>(0, GetSize()/3, 1, 0, 1e3))
	);
}

TEST_P(PatchCompressionTest, CompressionOf_PatternA_Char)
{
	OutsidePatchEncoding encoder;
	EXPECT_TRUE(
		TestContent<char>(
			boost::bind(&OutsidePatchEncoding::Encode<char>, encoder, _1),
			boost::bind(&OutsidePatchEncoding::Decode<char>, encoder, _1),
			GetFakeDataWithPatternA<char>(0, GetSize()/3, 1, 0, 1e2))
	);
	LowerPatchEncoding encoder2;
	EXPECT_TRUE(
		TestContent<char>(
			boost::bind(&LowerPatchEncoding::Encode<char>, encoder2, _1),
			boost::bind(&LowerPatchEncoding::Decode<char>, encoder2, _1),
			GetFakeDataWithPatternA<char>(0, GetSize()/3, 1, 0, 1e2))
	);
}


} /* namespace ddj */
