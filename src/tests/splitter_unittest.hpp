#ifndef DDJ_UTIL_SPLITTER_UNITTEST_HPP_
#define DDJ_UTIL_SPLITTER_UNITTEST_HPP_

#include "helpers/helper_device.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "core/cuda_ptr.hpp"

#include <gtest/gtest.h>

namespace ddj {

class SplitterTest : public testing::Test
{
protected:
	SplitterTest() : size(10000)
    {
		HelperDevice hc;
        hc.SetCudaDeviceWithMaxFreeMem();
    }

    ~SplitterTest(){}

	virtual void SetUp()
	{
		int n = size;
		d_float_random_data = generator.GenerateRandomFloatDeviceArray(n);
		d_int_random_data = generator.GenerateRandomIntDeviceArray(n);
	}

	SharedCudaPtr<float> d_float_random_data;
	SharedCudaPtr<int> d_int_random_data;
	const int size;

private:
	CudaArrayGenerator generator;
};

} /* namespace ddj */
#endif /* DDJ_UTIL_SPLITTER_UNITTEST_HPP_ */
