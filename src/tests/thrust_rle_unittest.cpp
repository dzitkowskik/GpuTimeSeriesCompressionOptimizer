#include "thrust_rle_unittest.h"

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
        d_random_data,
        real_size,
        compressed_size);

    void* decompressedData = compression.Decode(
        compressedData,
        compressed_size,
        decompressed_size);

    EXPECT_EQ(real_size, decompressed_size);
}


} /* namespace ddj */
