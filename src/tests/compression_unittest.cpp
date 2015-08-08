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
#include "compression/scale/scale_encoding.hpp"
#include "compression/delta/delta_encoding.hpp"
#include "compression/afl/afl_encoding.hpp"

#include <thrust/device_ptr.h>
#include <boost/bind.hpp>

namespace ddj
{

INSTANTIATE_TEST_CASE_P(
    ScaleEncoding_Compression_Inst,
    ScaleCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(ScaleCompressionTest, CompressionOfRandomInts_size)
{
    ScaleEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize<int>(
		boost::bind(&ScaleEncoding::Encode<int>, encoder, _1),
		boost::bind(&ScaleEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
    );
}

TEST_P(ScaleCompressionTest, CompressionOfRandomInts_data)
{
	ScaleEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent<int>(
		boost::bind(&ScaleEncoding::Encode<int>, encoder, _1),
		boost::bind(&ScaleEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
	);
}

TEST_P(ScaleCompressionTest, CompressionOfRandomFloats_size)
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

TEST_P(DeltaCompressionTest, CompressionOfRandomInts_size)
{
	DeltaEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize<int>(
		boost::bind(&DeltaEncoding::Encode<int>, encoder, _1),
		boost::bind(&DeltaEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
    );
}

TEST_P(DeltaCompressionTest, CompressionOfRandomInts_data)
{
	DeltaEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent<int>(
		boost::bind(&DeltaEncoding::Encode<int>, encoder, _1),
		boost::bind(&DeltaEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
	);
}

TEST_P(DeltaCompressionTest, CompressionOfRandomFloats_size)
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

INSTANTIATE_TEST_CASE_P(
	AflEncoding_Compression_Inst,
	AflCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(AflCompressionTest, CompressionOfRandomInts_size)
{
	AflEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize<int>(
		boost::bind(&AflEncoding::Encode<int>, encoder, _1),
		boost::bind(&AflEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
    );
}

TEST_P(AflCompressionTest, CompressionOfRandomInts_data)
{
	AflEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent<int>(
		boost::bind(&AflEncoding::Encode<int>, encoder, _1),
		boost::bind(&AflEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
	);
}


} /* namespace ddj */
