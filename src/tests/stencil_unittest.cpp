#include "stencil_unittest.hpp"
#include "helpers/helper_comparison.cuh"
#include "util/stencil/stencil.hpp"
#include "helpers/helper_print.hpp"

namespace ddj {

TEST_F(StencilTest, Stencil_random_pack_size)
{
    Stencil stencil(d_random_stencil_data);
    auto packed = stencil.pack();
    EXPECT_TRUE(packed->size() < stencil->size());
}

TEST_F(StencilTest, Stencil_random_pack_unpack_size)
{
    Stencil stencil(d_random_stencil_data);
    auto packed = stencil.pack();
    auto unpacked = stencil.unpack(packed);
    EXPECT_EQ(stencil->size(), unpacked->size());
}

TEST_F(StencilTest, Stencil_random_pack_unpack_data)
{
    Stencil stencil(d_random_stencil_data);
    auto packed = stencil.pack();
    auto unpacked = stencil.unpack(packed);
    auto result = CompareDeviceArrays(stencil->get(), unpacked->get(), stencil->size());
    EXPECT_TRUE(result);
}

} /* namespace ddj */
