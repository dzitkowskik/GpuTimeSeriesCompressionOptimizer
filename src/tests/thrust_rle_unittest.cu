#include "thrust_rle_unittest.hpp"
#include "helpers/helper_comparison.cuh"

namespace ddj {

INSTANTIATE_TEST_CASE_P(
    RandomFloatNumbersCompression_ThrustRle_Inst,
    ThrustRleCompressionTest,
    ::testing::Values(10, 20));

TEST_P(ThrustRleCompressionTest, CompressionOfRandomFloats_size)
{
    int real_size = GetParam();
    int compressed_size;
    int decompressed_size;

    void* compressedData = compression.Encode(
        d_random_data->get(),
        real_size,
        compressed_size);

    float* decompressedData = compression.Decode<float>(
        compressedData,
        compressed_size,
        decompressed_size);

    EXPECT_EQ(real_size, decompressed_size);

    CUDA_CALL(cudaFree(compressedData));
    CUDA_CALL(cudaFree(decompressedData));
}

TEST_P(ThrustRleCompressionTest, CompressionOfRandomFloats_data)
{
    int real_size = GetParam();
    int compressed_size;
    int decompressed_size;

    void* compressedData = compression.Encode(
        d_random_data->get(),
        real_size,
        compressed_size);

    float* decompressedData = compression.Decode<float>(
        compressedData,
        compressed_size,
        decompressed_size);

    EXPECT_TRUE(CompareDeviceArrays(
        d_random_data->get(), (float*)decompressedData, real_size));

    CUDA_CALL(cudaFree(compressedData));
    CUDA_CALL(cudaFree(decompressedData));
}

} /* namespace ddj */
