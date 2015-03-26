#include "helper_comparison_unittest.h"
#include "../helpers/helper_comparison.cuh"

namespace ddj {

INSTANTIATE_TEST_CASE_P(
    HelperComparisonTest_RandomFloats_Inst,
    HelperComparisonTest,
    ::testing::Values(100, 2000));

TEST_P(HelperComparisonTest, ComparisonOfRandomFloats_Equal)
{
    int size = GetParam();
    EXPECT_TRUE(CompareDeviceArrays(d_random_data, d_random_data, size));
}

TEST_P(HelperComparisonTest, ComparisonOfRandomFloats_NotEqual)
{
    int size = GetParam();
    EXPECT_FALSE(CompareDeviceArrays(d_random_data, d_random_data_2, size));
}

} /* namespace ddj */
