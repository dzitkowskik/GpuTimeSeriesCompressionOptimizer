#include "tests/patch_unittest.hpp"
#include "compression/patch/patch.cuh"
#include "compression/operators.cuh"

namespace ddj {

TEST_F(PatchedDataTest, Patch_init_noexcept)
{
    outsideOperator<int> op{500, 5000};
    PatchedData<int, outsideOperator<int>> patch(op);
    auto data = CudaPtr<int>::make_shared(size);
    patch.Init(d_int_random_data);
}

}
