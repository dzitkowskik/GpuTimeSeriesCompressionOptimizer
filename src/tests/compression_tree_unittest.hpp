/*
 *  compression_tree_unittest.hpp
 *
 *  Created on: 15/10/2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_COMPRESSION_TREE_UNITTEST_HPP_
#define DDJ_COMPRESSION_TREE_UNITTEST_HPP_

#include "helpers/helper_device.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "core/cuda_ptr.hpp"
#include <gtest/gtest.h>

namespace ddj
{

class CompressionTreeTest : public testing::Test,
	public ::testing::WithParamInterface<int>
{
public:
	CompressionTreeTest()
	{
		HelperDevice hc;
		hc.SetCudaDeviceWithMaxFreeMem();
	}
	virtual ~CompressionTreeTest() {}

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

} /* namespace ddj */
#endif /* DDJ_COMPRESSION_TREE_UNITTEST_HPP_ */
