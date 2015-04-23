/*
 * delta_unittest.cu
 *
 *  Created on: 22-04-2015
 *      Author: Karol Dzitkowski
 */

#include "delta_unittest.h"
#include "helpers/helper_comparison.cuh"
#include <thrust/device_ptr.h>
#include "helpers/helper_print.h"

namespace ddj
{

INSTANTIATE_TEST_CASE_P(
    RandomFloatNumbersEncoding_Delta_Inst,
    DeltaEncodingTest,
    ::testing::Values(10, 20));

TEST_P(DeltaEncodingTest, EncodingOfRandomFloats_size)
{
    int real_size = GetParam();
    int compressed_size;
    int decompressed_size;

    DeltaEncodingMetadata<float> metadata;

    void* compressedData = compression.Encode(
        d_random_data,
        real_size,
        compressed_size,
        metadata);

    float* decompressedData = compression.Decode<float>(
        compressedData,
        compressed_size,
        decompressed_size,
        metadata);

    EXPECT_EQ(real_size-1, compressed_size);
    EXPECT_EQ(real_size, decompressed_size);

    CUDA_CALL(cudaFree(compressedData));
    CUDA_CALL(cudaFree(decompressedData));
}

TEST_P(DeltaEncodingTest, CompressionOfRandomFloats_data)
{
    int real_size = GetParam();
    int compressed_size;
    int decompressed_size;

    DeltaEncodingMetadata<float> metadata;

    void* compressedData = compression.Encode(
        d_random_data,
        real_size,
        compressed_size,
        metadata);

    float* decompressedData = compression.Decode<float>(
        compressedData,
        compressed_size,
        decompressed_size,
        metadata);

    thrust::device_ptr<float> p1(d_random_data);
    thrust::device_ptr<float> p2(decompressedData);
    thrust::device_ptr<float> p3((float*)compressedData);

//    HelperPrint::PrintDevicePtr(p1, real_size, "Expected");
//    HelperPrint::PrintDevicePtr(p2, real_size, "Actual");
//    HelperPrint::PrintDevicePtr(p3, real_size-1, "Compressed");

    EXPECT_TRUE(CompareDeviceFloatArrays(d_random_data, decompressedData, real_size));

    CUDA_CALL(cudaFree(compressedData));
    CUDA_CALL(cudaFree(decompressedData));
}

} /* namespace ddj */
