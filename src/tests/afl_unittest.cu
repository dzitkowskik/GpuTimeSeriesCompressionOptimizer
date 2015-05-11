#include "afl_unittest.hpp"
#include "helpers/helper_comparison.cuh"

namespace ddj
{

INSTANTIATE_TEST_CASE_P(
    RandomFloatNumbersCompression_Afl_Inst,
    AflCompressionTest,
    ::testing::Values(10, 20));

TEST_P(AflCompressionTest, CompressionOfRandomInt_size)
{
    int real_size = GetParam();
    int compressed_size;
    int decompressed_size;
    AFLCompressionMetadata metadata;
    void* compressedData = compression.Encode(
        d_random_data,
        real_size,
        compressed_size,
        metadata);

    void* decompressedData = compression.Decode(
        (int*)compressedData,
        compressed_size,
        decompressed_size,
        metadata);

    EXPECT_EQ(real_size, decompressed_size);

    CUDA_CALL(cudaFree(compressedData));
    CUDA_CALL(cudaFree(decompressedData));
}

TEST_P(AflCompressionTest, CompressionOfRandomInt_data)
{
    int real_size = GetParam();
    int compressed_size;
    int decompressed_size;
    AFLCompressionMetadata metadata;
    void* compressedData = compression.Encode(
        d_random_data,
        real_size,
        compressed_size,
        metadata);

    int* decompressedData = (int*)compression.Decode(
        (int*)compressedData,
        compressed_size,
        decompressed_size,
        metadata);

    EXPECT_TRUE(CompareDeviceArrays(d_random_data, decompressedData, real_size));

    CUDA_CALL(cudaFree(compressedData));
    CUDA_CALL(cudaFree(decompressedData));
}

} /* namespace ddj */
