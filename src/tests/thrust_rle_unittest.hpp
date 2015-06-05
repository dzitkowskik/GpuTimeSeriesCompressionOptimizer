#ifndef DDJ_THRUST_RLE_UNITTEST_H_
#define DDJ_THRUST_RLE_UNITTEST_H_

#include "compression/rle/thrust_rle.cuh"
#include "helpers/helper_cuda.hpp"
#include "helpers/helper_macros.h"
#include "helpers/helper_generator.hpp"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>

namespace ddj {

class ThrustRleCompressionTest : public testing::Test,
    public ::testing::WithParamInterface<int>
{
protected:
    ThrustRleCompressionTest()
    {
        HelperCuda hc;
        hc.SetCudaDeviceWithMaxFreeMem();
        d_random_data = NULL;
    }

    virtual ~ThrustRleCompressionTest(){}

    virtual void SetUp()
    {
        int n = GetParam();
        d_random_data = generator.GenerateRandomFloatDeviceArray(n);
    }

    SharedCudaPtr<float> d_random_data;
    ThrustRleCompression compression;

private:
    HelperGenerator generator;
};

} /* namespace ddj */
#endif
