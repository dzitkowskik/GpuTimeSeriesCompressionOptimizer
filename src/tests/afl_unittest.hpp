/*
 *  afl_unittest.hpp
 *
 *  Created on: 22-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_GPUVEC_UNITTEST_H_
#define DDJ_GPUVEC_UNITTEST_H_

#include "compression/afl/afl.cuh"
#include "helpers/helper_cuda.hpp"
#include "helpers/helper_macros.h"
#include "helpers/helper_generator.hpp"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda.h>

namespace ddj {

class AflCompressionTest : public testing::Test,
    public ::testing::WithParamInterface<int>
{
protected:
	AflCompressionTest() : d_random_data(NULL)
    {
        HelperCuda hc;
        hc.SetCudaDeviceWithMaxFreeMem();
    }

    ~AflCompressionTest(){}

    virtual void SetUp()
    {
        int n = GetParam();
        d_random_data = generator.GenerateRandomIntDeviceArray(n);
    }

    virtual void TearDown()
    {
        CUDA_CALL(cudaFree(d_random_data));
    }

    int* d_random_data;
    AFLCompression compression;

private:
    HelperGenerator generator;
};

} /* namespace ddj */
#endif
