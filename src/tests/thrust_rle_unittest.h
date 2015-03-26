#ifndef DDJ_GPUVEC_UNITTEST_H_
#define DDJ_GPUVEC_UNITTEST_H_

#include "../compression/rle/thrust_rle.cuh"
#include "../helpers/helper_cuda.h"
#include "../helpers/helper_macros.h"

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
        CUDA_CALL(cudaMalloc((void**)&d_random_data, n * sizeof(float)));
        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1991ULL));
        CURAND_CALL(curandGenerateUniform(gen, d_random_data, n));
        CURAND_CALL(curandDestroyGenerator(gen));
    }

    virtual void TearDown()
    {
        CUDA_CALL(cudaFree(d_random_data));
    }

    float* d_random_data;
    ThrustRleCompression compression;
};

} /* namespace ddj */
#endif

