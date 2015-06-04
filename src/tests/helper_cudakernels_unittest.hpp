#ifndef DDJ_HELPER_CUDAKERNELS_UNITTEST_H_
#define DDJ_HELPER_CUDAKERNELS_UNITTEST_H_

#include "helpers/helper_cuda.hpp"
#include "helpers/helper_generator.hpp"
#include "core/cuda_ptr.hpp"
#include "helpers/helper_cudakernels.cuh"
#include <gtest/gtest.h>

namespace ddj {

class HelperCudaKernelsTest : public testing::Test
{
protected:
	HelperCudaKernelsTest() : size(10000)
    {
        HelperCuda hc;
        hc.SetCudaDeviceWithMaxFreeMem();
    }

    ~HelperCudaKernelsTest(){}

	virtual void SetUp()
	{
		int n = size;
		d_float_random_data = CudaPtr<float>::make_shared(
				generator.GenerateRandomFloatDeviceArray(n), n);
		d_int_random_data = CudaPtr<int>::make_shared(
				generator.GenerateRandomIntDeviceArray(n), n);
	}

	SharedCudaPtr<float> d_float_random_data;
	SharedCudaPtr<int> d_int_random_data;
	const int size;
	HelperCudaKernels kernels;
private:
	HelperGenerator generator;
};

} /* namespace ddj */
#endif
