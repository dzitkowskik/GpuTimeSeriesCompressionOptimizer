/*
 * delta_unittest.hpp
 *
 *  Created on: 22-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_COMPRESSION_UNITTEST_HPP_
#define DDJ_COMPRESSION_UNITTEST_HPP_

#include "helpers/helper_device.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "core/cuda_ptr.hpp"
#include <gtest/gtest.h>

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
#endif /* DDJ_COMPRESSION_UNITTEST_HPP_ */
