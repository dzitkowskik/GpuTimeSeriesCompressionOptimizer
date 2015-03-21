#ifndef DDJ_GPUVEC_UNITTEST_H_
#define DDJ_GPUVEC_UNITTEST_H_

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include "../compression/rle/thrust_rle.cuh"
#include "../cudacommons.h"
#include "../helper_cuda.h"

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);}} while(0)

namespace ddj {

class ThrustRleCompressionTest : public testing::Test,
    public ::testing::WithParamInterface<int>
{
protected:
    ThrustRleCompressionTest()
    {
        CudaCommons cudaC;
        cudaC.SetCudaDeviceWithMaxFreeMem();
    }

    ~ThrustRleCompressionTest(){}

    virtual void SetUp()
    {
        int n = GetParam();

        // set up random data
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

