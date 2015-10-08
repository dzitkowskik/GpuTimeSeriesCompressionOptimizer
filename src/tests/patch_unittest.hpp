/*
 *  patch_unittest.hpp
 *
 *  Created on: 04-06-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_PATCH_UNITTEST_H_
#define DDJ_PATCH_UNITTEST_H_

#include "helpers/helper_device.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "core/cuda_ptr.hpp"
#include "helpers/helper_cudakernels.cuh"

#include <gtest/gtest.h>

namespace ddj {

class PatchedDataTest : public testing::Test
{
protected:
    PatchedDataTest() : size(10000)
    {
        HelperDevice hc;
        hc.SetCudaDeviceWithMaxFreeMem();
    }

    ~PatchedDataTest(){}

	virtual void SetUp()
	{
		int n = size;
		d_float_random_data = generator.GenerateRandomFloatDeviceArray(n);
		d_int_random_data = generator.GenerateRandomIntDeviceArray(n);
        d_int_consecutive_data = generator.GenerateConsecutiveIntDeviceArray(n);
	}

	SharedCudaPtr<float> d_float_random_data;
	SharedCudaPtr<int> d_int_random_data;
    SharedCudaPtr<int> d_int_consecutive_data;
	const int size;

private:
	CudaArrayGenerator generator;
};

} /* namespace ddj */
#endif //DDJ_PATCH_UNITTEST_H_
