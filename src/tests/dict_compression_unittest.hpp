/*
 *  dict_compression_unittest.hpp
 *
 *  Created on: 01-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_DICT_COMPRESSION_UNITTEST_HPP_
#define DDJ_DICT_COMPRESSION_UNITTEST_HPP_

#include "helpers/helper_device.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "core/cuda_ptr.hpp"

#include <gtest/gtest.h>

namespace ddj {

class DictCompressionTest : public testing::Test,
    public ::testing::WithParamInterface<int>
{
protected:
	DictCompressionTest() : size(10000)
    {
		HelperDevice hc;
        hc.SetCudaDeviceWithMaxFreeMem();
    }

	virtual void SetUp()
	{
		int n = size;
		d_int_random_data = generator.GenerateRandomIntDeviceArray(n, 100, 1000);
	}

	SharedCudaPtr<int> d_int_random_data;
	const int size;

private:
	CudaArrayGenerator generator;
};

} /* namespace ddj */
#endif /* DDJ_DICT_COMPRESSION_UNITTEST_HPP_ */
