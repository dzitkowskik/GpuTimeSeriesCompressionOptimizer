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
    ::testing::Values(10, 1000, 10000));

TEST_P(PatchCompressionTest, OutsidePatch_CompressionOfRandomInts_size)
{
	OutsidePatchEncoding encoder(501, 5000);
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&OutsidePatchEncoding::Encode<int>, encoder, _1),
			boost::bind(&OutsidePatchEncoding::Decode<int>, encoder, _1),
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
}

TEST_P(PatchCompressionTest, OutsidePatch_CompressionOfRandomInts_bigData)
{
	OutsidePatchEncoding encoder(501, 5000);
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&OutsidePatchEncoding::Encode<int>, encoder, _1),
			boost::bind(&OutsidePatchEncoding::Decode<int>, encoder, _1),
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
}


} /* namespace ddj */
