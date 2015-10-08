/*
 *  histogram_unittest.hpp
 *
 *  Created on: 05-07-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_HISTOGRAM_UNITTEST_HPP_
#define DDJ_HISTOGRAM_UNITTEST_HPP_

#include "helpers/helper_device.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "core/cuda_ptr.hpp"
#include <boost/function.hpp>
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
		d_int_random_data = generator.GenerateRandomIntDeviceArray(n, 100, 1000);
	}

	SharedCudaPtr<float> d_float_random_data;
	SharedCudaPtr<int> d_int_random_data;
	const int size;

protected:
	template<typename T>
	void CheckHistogramResult(SharedCudaPtr<T> data, SharedCudaPtrPair<T, int> result);

private:
	CudaArrayGenerator generator;
};

} /* namespace ddj */
#endif /* DDJ_HISTOGRAM_UNITTEST_HPP_ */
