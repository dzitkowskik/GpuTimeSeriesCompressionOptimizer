#ifndef DDJ_HELPER_COMPARISON_UNITTEST_H_
#define DDJ_HELPER_COMPARISON_UNITTEST_H_

#include "../helpers/helper_macros.h"
#include "../helpers/helper_comparison.cuh"
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
    HelperComparisonTest()
    {
        HelperCuda hc;
        hc.SetCudaDeviceWithMaxFreeMem();
    }

    ~HelperComparisonTest(){}

    virtual void SetUp()
    {
        int n = GetParam();
        curandGenerator_t gen;
        CUDA_CALL(cudaMalloc((void**)&d_random_data, n * sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_random_data_2, n * sizeof(float)));
        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1991ULL));
        CURAND_CALL(curandGenerateUniform(gen, d_random_data, n));
        CURAND_CALL(curandGenerateUniform(gen, d_random_data_2, n));
        CURAND_CALL(curandDestroyGenerator(gen));
    }

    virtual void TearDown()
    {
        CUDA_CALL(cudaFree(d_random_data));
    }

    float* d_random_data;
    float* d_random_data_2;
};

} /* namespace ddj */

#endif /* DDJ_HELPER_COMPARISON_UNITTEST_H_ */
