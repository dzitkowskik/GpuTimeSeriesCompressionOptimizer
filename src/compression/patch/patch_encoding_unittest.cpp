#include "test/compression_unittest_base.hpp"
#include "patch_encoding.hpp"
#include "util/stencil/stencil_operators.hpp"
#include <gtest/gtest.h>
#include <boost/bind.hpp>

namespace ddj {

class PatchCompressionTest : public CompressionUnittestBase {};
class PatchTest : public UnittestBase {};

INSTANTIATE_TEST_CASE_P(
	PatchEncoding_Compression_Inst,
	PatchCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_F(PatchTest, Patch_encode_size)
{
    OutsideOperator<int> op{501, 5000};
    PatchEncoding<OutsideOperator<int>> patch(op);
    auto result = patch.Encode(GetIntConsecutiveData());

    EXPECT_EQ(4500*sizeof(int), result[1]->size());
    EXPECT_EQ(5500*sizeof(int), result[2]->size());
}

TEST_P(PatchCompressionTest, CompressionOfRandomInts_size)
{
	OutsideOperator<int> op{501, 5000};
	PatchEncoding<OutsideOperator<int>> encoder(op);
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&PatchEncoding<OutsideOperator<int>>::Encode<int>, encoder, _1),
			boost::bind(&PatchEncoding<OutsideOperator<int>>::Decode<int>, encoder, _1),
			GetIntRandomData())
    );
}

TEST_P(PatchCompressionTest, CompressionOfRandomInts_data)
{
	OutsideOperator<int> op{501, 5000};
	PatchEncoding<OutsideOperator<int>> encoder(op);
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&PatchEncoding<OutsideOperator<int>>::Encode<int>, encoder, _1),
			boost::bind(&PatchEncoding<OutsideOperator<int>>::Decode<int>, encoder, _1),
			GetIntRandomData())
	);
}

TEST_P(PatchCompressionTest, CompressionOfRandomFloats_size)
{
	OutsideOperator<float> op{501.0f, 5000.0f};
	PatchEncoding<OutsideOperator<float>> encoder(op);
    EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&PatchEncoding<OutsideOperator<float>>::Encode<float>, encoder, _1),
			boost::bind(&PatchEncoding<OutsideOperator<float>>::Decode<float>, encoder, _1),
			GetFloatRandomData())
    );
}

TEST_P(PatchCompressionTest, CompressionOfRandomFloats_data)
{
	OutsideOperator<float> op{501.0f, 5000.0f};
	PatchEncoding<OutsideOperator<float>> encoder(op);
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&PatchEncoding<OutsideOperator<float>>::Encode<float>, encoder, _1),
			boost::bind(&PatchEncoding<OutsideOperator<float>>::Decode<float>, encoder, _1),
			GetFloatRandomData())
	);
}


} /* namespace ddj */
