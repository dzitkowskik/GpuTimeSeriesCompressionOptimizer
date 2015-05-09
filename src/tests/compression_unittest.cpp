/*
 * delta_unittest.cu
 *
 *  Created on: 22-04-2015
 *      Author: Karol Dzitkowski
 */

#include "compression_unittest.h"
#include "helpers/helper_comparison.cuh"
#include "helpers/helper_print.h"
#include "encode_decode_unittest_helper.h"
#include "compression/scale/scale_encoding.cuh"

#include <thrust/device_ptr.h>
#include <boost/bind.hpp>

namespace ddj
{

INSTANTIATE_TEST_CASE_P(
    ScaleEncoding_Compression_Inst,
    CompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(CompressionTest, EncodingOfRandomFloats_size)
{
    ScaleEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize<float>(
		boost::bind(&ScaleEncoding::Encode<float>, encoder, _1),
		boost::bind(&ScaleEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
    );
}

TEST_P(CompressionTest, CompressionOfRandomFloats_data)
{
	ScaleEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent<float>(
		boost::bind(&ScaleEncoding::Encode<float>, encoder, _1),
		boost::bind(&ScaleEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
	);
}

} /* namespace ddj */
