#ifndef DDJ_HELPER_CUDAKERNELS_UNITTEST_H_
#define DDJ_HELPER_CUDAKERNELS_UNITTEST_H_

#include "helpers/helper_cuda.hpp"
#include <gtest/gtest.h>

namespace ddj {

class HelperCudaKernelsTest : public testing::Test
{
protected:
	HelperCudaKernelsTest()
    {
        HelperCuda hc;
        hc.SetCudaDeviceWithMaxFreeMem();
    }

    ~HelperCudaKernelsTest(){}
};

} /* namespace ddj */
#endif
