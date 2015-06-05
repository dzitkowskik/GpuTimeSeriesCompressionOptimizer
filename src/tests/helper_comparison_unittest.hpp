#ifndef DDJ_HELPER_COMPARISON_UNITTEST_H_
#define DDJ_HELPER_COMPARISON_UNITTEST_H_

#include "helpers/helper_macros.h"
#include "helpers/helper_comparison.cuh"
#include "helpers/helper_generator.hpp"
#include "helpers/helper_cuda.hpp"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>

namespace ddj {

class HelperComparisonTest : public testing::Test,
    public ::testing::WithParamInterface<int>
{
protected:
    HelperComparisonTest() : d_random_data(NULL), d_random_data_2(NULL)
    {
        HelperCuda hc;
        hc.SetCudaDeviceWithMaxFreeMem();
    }

    ~HelperComparisonTest(){}

    virtual void SetUp()
    {
        int n = GetParam();
        d_random_data = generator.GenerateRandomFloatDeviceArray(n);
        d_random_data_2 = generator.GenerateRandomFloatDeviceArray(n);
    }

    SharedCudaPtr<float> d_random_data;
    SharedCudaPtr<float> d_random_data_2;

private:
    HelperGenerator generator;
};

} /* namespace ddj */
#endif /* DDJ_HELPER_COMPARISON_UNITTEST_H_ */
