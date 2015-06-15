/*
 * delta_unittest.cu
 *
 *  Created on: 22-04-2015
 *      Author: Karol Dzitkowski
 */

#include "compression_unittest.hpp"
#include "helpers/helper_comparison.cuh"
#include "helpers/helper_print.hpp"
#include "encode_decode_unittest_helper.hpp"
#include "compression/scale/scale_encoding.cuh"
#include "compression/delta/delta_encoding.cuh"

#include <thrust/device_ptr.h>
#include <boost/bind.hpp>

namespace ddj
{

INSTANTIATE_TEST_CASE_P(
    ScaleEncoding_Compression_Inst,
    ScaleCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(ScaleCompressionTest, EncodingOfRandomFloats_size)
{
    ScaleEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize<float>(
		boost::bind(&ScaleEncoding::Encode<float>, encoder, _1),
		boost::bind(&ScaleEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
    );
}

TEST_P(ScaleCompressionTest, CompressionOfRandomFloats_data)
{
	ScaleEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent<float>(
		boost::bind(&ScaleEncoding::Encode<float>, encoder, _1),
		boost::bind(&ScaleEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
	);
}

INSTANTIATE_TEST_CASE_P(
	DeltaEncoding_Compression_Inst,
	DeltaCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(DeltaCompressionTest, EncodingOfRandomFloats_size)
{
	DeltaEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize<float>(
		boost::bind(&DeltaEncoding::Encode<float>, encoder, _1),
		boost::bind(&DeltaEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
    );
}

TEST_P(DeltaCompressionTest, CompressionOfRandomFloats_data)
{
	DeltaEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent<float>(
		boost::bind(&DeltaEncoding::Encode<float>, encoder, _1),
		boost::bind(&DeltaEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
	);
}

} /* namespace ddj */
