/*
 * delta_unittest.h
 *
 *  Created on: 22-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_DELTA_UNITTEST_H_
#define DDJ_DELTA_UNITTEST_H_

#include "helpers/helper_cuda.h"
#include "helpers/helper_macros.h"
#include "helpers/helper_generator.h"
#include "helpers/helper_print.h"
#include <thrust/device_vector.h>
#include "core/cuda_ptr.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>

namespace ddj
{

class CompressionTest : public testing::Test,
    public ::testing::WithParamInterface<int>
{
public:
	CompressionTest()
	{
		HelperCuda hc;
		hc.SetCudaDeviceWithMaxFreeMem();
	}

	virtual void SetUp()
	{
		int n = GetParam();
		d_float_random_data = CudaPtr<float>::make_shared(
				generator.GenerateRandomFloatDeviceArray(n), n);
		d_int_random_data = CudaPtr<int>::make_shared(
				generator.GenerateRandomIntDeviceArray(n), n);
	}

	SharedCudaPtr<float> d_float_random_data;
	SharedCudaPtr<int> d_int_random_data;

private:
	HelperGenerator generator;
};

class ScaleCompressionTest : public CompressionTest {};
class DeltaCompressionTest : public CompressionTest {};

} /* namespace ddj */
#endif /* DDJ_DELTA_UNITTEST_H_ */
