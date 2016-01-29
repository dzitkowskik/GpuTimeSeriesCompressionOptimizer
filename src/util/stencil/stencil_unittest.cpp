#include "test/unittest_base.hpp"
#include "core/cuda_array.hpp"

#include "stencil.hpp"

namespace ddj {

class StencilTest : public UnittestBase {};

TEST_F(StencilTest, Stencil_random_pack_size)
{
    Stencil stencil(GetRandomStencilData());
    auto packed = stencil.pack();
    EXPECT_TRUE(packed->size() < stencil->size());
}

TEST_F(StencilTest, Stencil_random_pack_unpack_size)
{
    Stencil stencil(GetRandomStencilData());
    auto packed = stencil.pack();
    auto unpacked = stencil.unpack(packed);
    EXPECT_EQ(stencil->size(), unpacked->size());
}

TEST_F(StencilTest, Stencil_random_pack_unpack_data)
{
    Stencil stencil(GetRandomStencilData());
    auto packed = stencil.pack();
    auto unpacked = stencil.unpack(packed);
    auto result = CompareDeviceArrays(stencil->get(), unpacked->get(), stencil->size());
    EXPECT_TRUE(result);
}

} /* namespace ddj */
