#include "tests/patch_unittest.hpp"
#include "helpers/helper_comparison.cuh"
#include "compression/patch/patch_encoding.hpp"
#include "core/operators.cuh"
#include "helpers/helper_print.hpp"

namespace ddj {

TEST_F(PatchedDataTest, Patch_encode_noexcept)
{
    OutsideOperator<int> op{500, 5000};
    PatchEncoding<OutsideOperator<int>> patch(op);
    patch.Encode(d_int_random_data);
}

TEST_F(PatchedDataTest, Patch_encode_size)
{
    OutsideOperator<int> op{501, 5000};
    PatchEncoding<OutsideOperator<int>> patch(op);
    auto result = patch.Encode(d_int_consecutive_data);

    EXPECT_EQ(4500*sizeof(int), result[1]->size());
    EXPECT_EQ(5500*sizeof(int), result[2]->size());
}

TEST_F(PatchedDataTest, Patch_init_Encode_Decode_size)
{
    OutsideOperator<int> op{501, 5000};
    PatchEncoding<OutsideOperator<int>> patch(op);
    auto encoded = patch.Encode<int>(d_int_consecutive_data);
    auto decoded = patch.Decode<int>(encoded);

    EXPECT_EQ(decoded->size(), d_int_consecutive_data->size());
}

TEST_F(PatchedDataTest, Patch_init_Encode_Decode_data)
{
    OutsideOperator<int> op{501, 5000};
    PatchEncoding<OutsideOperator<int>> patch(op);
    auto encoded = patch.Encode<int>(d_int_consecutive_data);
    auto decoded = patch.Decode<int>(encoded);

    EXPECT_TRUE(
        CompareDeviceArrays(
            d_int_consecutive_data->get(),
            decoded->get(),
            d_int_consecutive_data->size())
    );
}

}
