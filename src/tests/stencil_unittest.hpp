/*
 *  stencil_unittest.hpp
 *
 *  Created on: 04-06-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_STENCIL_UNITTEST_HPP_
#define DDJ_STENCIL_UNITTEST_HPP_

#include "helpers/helper_device.hpp"
#include "helpers/helper_generator.hpp"
#include "core/cuda_ptr.hpp"
#include <gtest/gtest.h>

namespace ddj {

class StencilTest : public testing::Test
{
protected:
    StencilTest() : size(10000)
    {
        HelperDevice hc;
        hc.SetCudaDeviceWithMaxFreeMem();
    }

    ~StencilTest(){}

	virtual void SetUp()
	{
		int n = size;
        d_random_stencil_data = generator.GenerateRandomStencil(n);
	}

    SharedCudaPtr<int> d_random_stencil_data;
	const int size;

private:
	HelperGenerator generator;
};

} /* namespace ddj */
#endif /* DDJ_STENCIL_UNITTEST_HPP_ */
