#ifndef DDJ_HELPER_CUDAKERNELS_UNITTEST_H_
#define DDJ_HELPER_CUDAKERNELS_UNITTEST_H_

#include "helpers/helper_device.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "core/cuda_ptr.hpp"
#include "helpers/helper_cudakernels.cuh"

#include <gtest/gtest.h>

namespace ddj {

class HelperCudaKernelsTest : public testing::Test
{
protected:
	HelperCudaKernelsTest() : size(10000)
    {
		HelperDevice hc;
        hc.SetCudaDeviceWithMaxFreeMem();
    }

    ~HelperCudaKernelsTest(){}

	virtual void SetUp()
	{
		int n = size;
		d_float_random_data = generator.GenerateRandomFloatDeviceArray(n);
		d_int_random_data = generator.GenerateRandomIntDeviceArray(n);
	}

	SharedCudaPtr<float> d_float_random_data;
	SharedCudaPtr<int> d_int_random_data;
	const int size;
	HelperCudaKernels kernels;

private:
	CudaArrayGenerator generator;
};

} /* namespace ddj */
#endif
