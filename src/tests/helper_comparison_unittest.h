#ifndef DDJ_HELPER_COMPARISON_UNITTEST_H_
#define DDJ_HELPER_COMPARISON_UNITTEST_H_

#include "../helpers/helper_macros.h"
#include "../helpers/helper_comparison.cuh"
#include "../helpers/helper_generator.h"
#include "../helpers/helper_cuda.h"

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

    virtual void TearDown()
    {
        CUDA_CALL(cudaFree(d_random_data));
        CUDA_CALL(cudaFree(d_random_data_2));
    }

    float* d_random_data;
    float* d_random_data_2;

private:
    HelperGenerator generator;
};

} /* namespace ddj */

#endif /* DDJ_HELPER_COMPARISON_UNITTEST_H_ */
