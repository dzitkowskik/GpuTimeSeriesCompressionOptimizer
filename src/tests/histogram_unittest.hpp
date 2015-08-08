/*
 *  histogram_unittest.hpp
 *
 *  Created on: 05-07-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_HISTOGRAM_UNITTEST_HPP_
#define DDJ_HISTOGRAM_UNITTEST_HPP_

#include "helpers/helper_device.hpp"
#include "helpers/helper_generator.hpp"
#include "core/cuda_ptr.hpp"
#include "util/histogram/histogram_base.hpp"
#include <gtest/gtest.h>

namespace ddj {

class HistogramTest : public testing::Test
{
protected:
	HistogramTest() : size(10000)
    {
		HelperDevice hc;
        hc.SetCudaDeviceWithMaxFreeMem();
    }

    ~HistogramTest(){}

	virtual void SetUp()
	{
		int n = size;
		d_float_random_data = generator.GenerateRandomFloatDeviceArray(n);
		d_int_random_data = generator.GenerateRandomIntDeviceArray(n);
	}

	SharedCudaPtr<float> d_float_random_data;
	SharedCudaPtr<int> d_int_random_data;
	const int size;

protected:
	void RandomIntegerArrayTestCase(HistogramBase& histogram);

private:
	HelperGenerator generator;
};

} /* namespace ddj */
#endif /* DDJ_HISTOGRAM_UNITTEST_HPP_ */
