/*
 * delta_unittest.hpp
 *
 *  Created on: 22-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_DELTA_UNITTEST_H_
#define DDJ_DELTA_UNITTEST_H_

#include "helpers/helper_device.hpp"
#include "helpers/helper_macros.h"
#include "util/generator/cuda_array_generator.hpp"
#include "helpers/helper_print.hpp"
#include <thrust/device_vector.h>
#include "core/cuda_ptr.hpp"

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
        HelperDevice hc;
		hc.SetCudaDeviceWithMaxFreeMem();
	}

	virtual void SetUp()
	{
		int n = GetParam();
		d_float_random_data = generator.GenerateRandomFloatDeviceArray(n);
		d_int_random_data = generator.GenerateRandomIntDeviceArray(n);
	}

	SharedCudaPtr<float> d_float_random_data;
	SharedCudaPtr<int> d_int_random_data;

private:
	CudaArrayGenerator generator;
};

class ScaleCompressionTest : public CompressionTest {};
class DeltaCompressionTest : public CompressionTest {};
class AflCompressionTest : public CompressionTest {};
class DictCompressionTest : public CompressionTest {};
class RleCompressionTest : public CompressionTest {};
class UniqueCompressionTest : public CompressionTest {};

} /* namespace ddj */
#endif /* DDJ_DELTA_UNITTEST_H_ */
