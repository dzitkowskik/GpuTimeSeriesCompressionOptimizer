/*
 *  afl_unittest.hpp
 *
 *  Created on: 22-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_GPUVEC_UNITTEST_H_
#define DDJ_GPUVEC_UNITTEST_H_

#include "compression/afl/afl.cuh"
#include "helpers/helper_device.hpp"
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
	AflCompressionTest()
    {
        HelperDevice hc;
        hc.SetCudaDeviceWithMaxFreeMem();
    }

    ~AflCompressionTest(){}

    virtual void SetUp()
    {
        int n = GetParam();
        d_random_data = generator.GenerateRandomIntDeviceArray(n);
    }

    SharedCudaPtr<int> d_random_data;
    AFLCompression compression;

private:
    HelperGenerator generator;
};

} /* namespace ddj */
#endif
