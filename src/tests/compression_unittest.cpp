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
#include "core/operators.cuh"
#include "compression/scale/scale_encoding.hpp"
#include "compression/delta/delta_encoding.hpp"
#include "compression/afl/afl_encoding.hpp"
#include "compression/dict/dict_encoding.hpp"
#include "compression/rle/rle_encoding.hpp"
#include "compression/unique/unique_encoding.hpp"
#include "compression/patch/patch_encoding.hpp"

#include <thrust/device_ptr.h>
#include <boost/bind.hpp>

namespace ddj
{

/////////////////////////
// SCALE COMPRESSION ////
/////////////////////////

INSTANTIATE_TEST_CASE_P(
    ScaleEncoding_Compression_Inst,
    ScaleCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(ScaleCompressionTest, CompressionOfRandomInts_size)
{
    ScaleEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize2<int>(
		boost::bind(&ScaleEncoding::Encode<int>, encoder, _1),
		boost::bind(&ScaleEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
    );
}

TEST_P(ScaleCompressionTest, CompressionOfRandomInts_data)
{
	ScaleEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent2<int>(
		boost::bind(&ScaleEncoding::Encode<int>, encoder, _1),
		boost::bind(&ScaleEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
	);
}

TEST_P(ScaleCompressionTest, CompressionOfRandomFloats_size)
{
    ScaleEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize2<float>(
		boost::bind(&ScaleEncoding::Encode<float>, encoder, _1),
		boost::bind(&ScaleEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
    );
}

TEST_P(ScaleCompressionTest, CompressionOfRandomFloats_data)
{
	ScaleEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent2<float>(
		boost::bind(&ScaleEncoding::Encode<float>, encoder, _1),
		boost::bind(&ScaleEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
	);
}

/////////////////////////
// DELTA COMPRESSION ////
/////////////////////////

INSTANTIATE_TEST_CASE_P(
	DeltaEncoding_Compression_Inst,
	DeltaCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(DeltaCompressionTest, CompressionOfRandomInts_size)
{
	DeltaEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize2<int>(
		boost::bind(&DeltaEncoding::Encode<int>, encoder, _1),
		boost::bind(&DeltaEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
    );
}

TEST_P(DeltaCompressionTest, CompressionOfRandomInts_data)
{
	DeltaEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent2<int>(
		boost::bind(&DeltaEncoding::Encode<int>, encoder, _1),
		boost::bind(&DeltaEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
	);
}

TEST_P(DeltaCompressionTest, CompressionOfRandomFloats_size)
{
	DeltaEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize2<float>(
		boost::bind(&DeltaEncoding::Encode<float>, encoder, _1),
		boost::bind(&DeltaEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
    );
}

TEST_P(DeltaCompressionTest, CompressionOfRandomFloats_data)
{
	DeltaEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent2<float>(
		boost::bind(&DeltaEncoding::Encode<float>, encoder, _1),
		boost::bind(&DeltaEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
	);
}

////////////////////////////////
// FIXED LENGTH COMPRESSION ////
////////////////////////////////

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

///////////////////////////////
// DICTIONARY COMPRESSION /////
///////////////////////////////

INSTANTIATE_TEST_CASE_P(
	DictEncoding_Compression_Inst,
	DictCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(DictCompressionTest, CompressionOfRandomInts_size)
{
	DictEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize2<int>(
		boost::bind(&DictEncoding::Encode<int>, encoder, _1),
		boost::bind(&DictEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
    );
}

TEST_P(DictCompressionTest, CompressionOfRandomInts_data)
{
	DictEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent2<int>(
		boost::bind(&DictEncoding::Encode<int>, encoder, _1),
		boost::bind(&DictEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
	);
}

TEST_P(DictCompressionTest, CompressionOfRandomFloats_size)
{
	DictEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize2<float>(
		boost::bind(&DictEncoding::Encode<float>, encoder, _1),
		boost::bind(&DictEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
    );
}

TEST_P(DictCompressionTest, CompressionOfRandomFloats_data)
{
	DictEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent2<float>(
		boost::bind(&DictEncoding::Encode<float>, encoder, _1),
		boost::bind(&DictEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
	);
}

////////////////////////////////
// RUN-LENGTH COMPRESSION //////
////////////////////////////////

INSTANTIATE_TEST_CASE_P(
	RleEncoding_Compression_Inst,
	RleCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(RleCompressionTest, CompressionOfRandomInts_size)
{
	RleEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize2<int>(
		boost::bind(&RleEncoding::Encode<int>, encoder, _1),
		boost::bind(&RleEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
    );
}

TEST_P(RleCompressionTest, CompressionOfRandomInts_data)
{
	RleEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent2<int>(
		boost::bind(&RleEncoding::Encode<int>, encoder, _1),
		boost::bind(&RleEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
	);
}

TEST_P(RleCompressionTest, CompressionOfRandomFloats_size)
{
	RleEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize2<float>(
		boost::bind(&RleEncoding::Encode<float>, encoder, _1),
		boost::bind(&RleEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
    );
}

TEST_P(RleCompressionTest, CompressionOfRandomFloats_data)
{
	RleEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent2<float>(
		boost::bind(&RleEncoding::Encode<float>, encoder, _1),
		boost::bind(&RleEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
	);
}

////////////////////////////
// UNIQUE COMPRESSION //////
////////////////////////////

INSTANTIATE_TEST_CASE_P(
	UniqueEncoding_Compression_Inst,
	UniqueCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(UniqueCompressionTest, CompressionOfRandomInts_size)
{
	UniqueEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize2<int>(
		boost::bind(&UniqueEncoding::Encode<int>, encoder, _1),
		boost::bind(&UniqueEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
    );
}

TEST_P(UniqueCompressionTest, CompressionOfRandomInts_data)
{
	UniqueEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent2<int>(
		boost::bind(&UniqueEncoding::Encode<int>, encoder, _1),
		boost::bind(&UniqueEncoding::Decode<int>, encoder, _1),
		d_int_random_data)
	);
}

TEST_P(UniqueCompressionTest, CompressionOfRandomFloats_size)
{
	UniqueEncoding encoder;
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize2<float>(
		boost::bind(&UniqueEncoding::Encode<float>, encoder, _1),
		boost::bind(&UniqueEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
    );
}

TEST_P(UniqueCompressionTest, CompressionOfRandomFloats_data)
{
	UniqueEncoding encoder;
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent2<float>(
		boost::bind(&UniqueEncoding::Encode<float>, encoder, _1),
		boost::bind(&UniqueEncoding::Decode<float>, encoder, _1),
		d_float_random_data)
	);
}

////////////////////////////
// PATCH COMPRESSION  //////
////////////////////////////

INSTANTIATE_TEST_CASE_P(
	PatchEncoding_Compression_Inst,
	PatchCompressionTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(PatchCompressionTest, CompressionOfRandomInts_size)
{
	OutsideOperator<int> op{501, 5000};
	PatchEncoding<OutsideOperator<int>> encoder(op);
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize2<int>(
		boost::bind(&PatchEncoding<OutsideOperator<int>>::Encode<int>, encoder, _1),
		boost::bind(&PatchEncoding<OutsideOperator<int>>::Decode<int>, encoder, _1),
		d_int_random_data)
    );
}

TEST_P(PatchCompressionTest, CompressionOfRandomInts_data)
{
	OutsideOperator<int> op{501, 5000};
	PatchEncoding<OutsideOperator<int>> encoder(op);
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent2<int>(
		boost::bind(&PatchEncoding<OutsideOperator<int>>::Encode<int>, encoder, _1),
		boost::bind(&PatchEncoding<OutsideOperator<int>>::Decode<int>, encoder, _1),
		d_int_random_data)
	);
}

TEST_P(PatchCompressionTest, CompressionOfRandomFloats_size)
{
	OutsideOperator<int> op{501, 5000};
	PatchEncoding<OutsideOperator<int>> encoder(op);
    EXPECT_TRUE(
    EncodeDecodeUnittestHelper::TestSize2<float>(
		boost::bind(&PatchEncoding<OutsideOperator<int>>::Encode<float>, encoder, _1),
		boost::bind(&PatchEncoding<OutsideOperator<int>>::Decode<float>, encoder, _1),
		d_float_random_data)
    );
}

TEST_P(PatchCompressionTest, CompressionOfRandomFloats_data)
{
	OutsideOperator<int> op{501, 5000};
	PatchEncoding<OutsideOperator<int>> encoder(op);
	EXPECT_TRUE(
	EncodeDecodeUnittestHelper::TestContent2<float>(
		boost::bind(&PatchEncoding<OutsideOperator<int>>::Encode<float>, encoder, _1),
		boost::bind(&PatchEncoding<OutsideOperator<int>>::Decode<float>, encoder, _1),
		d_float_random_data)
	);
}

} /* namespace ddj */
