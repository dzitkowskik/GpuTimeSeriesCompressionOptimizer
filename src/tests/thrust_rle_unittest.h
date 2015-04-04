#ifndef DDJ_GPUVEC_UNITTEST_H_
#define DDJ_GPUVEC_UNITTEST_H_

#include "../compression/rle/thrust_rle.cuh"
#include "../helpers/helper_cuda.h"
#include "../helpers/helper_macros.h"
#include "../helpers/helper_generator.h"

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
    }

    ~ThrustRleCompressionTest(){}

    virtual void SetUp()
    {
        int n = GetParam();
        curandGenerator_t gen;
        d_random_data = generator.GenerateRandomDeviceArray(n);
    }

    virtual void TearDown()
    {
        CUDA_CALL(cudaFree(d_random_data));
    }

    float* d_random_data;
    ThrustRleCompression compression;

private:
    HelperGenerator generator;
};

} /* namespace ddj */
#endif

