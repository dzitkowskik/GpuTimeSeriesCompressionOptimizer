#include "helper_comparison_unittest.hpp"
#include "core/cuda_array.hpp"

namespace ddj {

INSTANTIATE_TEST_CASE_P(
    HelperComparisonTest_RandomFloats_Inst,
    HelperComparisonTest,
    ::testing::Values(100, 2000, 10000));

TEST_P(HelperComparisonTest, ComparisonOfRandomFloats_Equal)
{
    int size = GetParam();
    EXPECT_TRUE( CompareDeviceArrays(d_random_data->get(), d_random_data->get(), size) );
}

TEST_P(HelperComparisonTest, ComparisonOfRandomFloats_NotEqual)
{
    int size = GetParam();
    EXPECT_FALSE( CompareDeviceArrays(d_random_data->get(), d_random_data_2->get(), size) );
}

} /* namespace ddj */
