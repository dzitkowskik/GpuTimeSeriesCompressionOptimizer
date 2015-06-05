#include "tests/patch_unittest.hpp"
#include "compression/patch/patch.cuh"
#include "compression/operators.cuh"
#include "helpers/helper_print.hpp"

namespace ddj {

TEST_F(PatchedDataTest, Patch_init_noexcept)
{
    OutsideOperator<int> op{500, 5000};
    PatchedData<int, OutsideOperator<int>> patch(op);
    auto data = CudaPtr<int>::make_shared(size);
    patch.Init(d_int_random_data);
}

TEST_F(PatchedDataTest, Patch_init_size)
{
    OutsideOperator<int> op{501, 5000};
    PatchedData<int, OutsideOperator<int>> patch(op);
    patch.Init(d_int_consecutive_data);

    auto first = patch.GetFirst();
    auto second = patch.GetSecond();

    EXPECT_EQ(4500, first->size());
    EXPECT_EQ(5500, second->size());
}

}
