#ifndef DDJ_GPUVEC_UNITTEST_H_
#define DDJ_GPUVEC_UNITTEST_H_

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "store/gpuvec.hpp"
#include "helpers/helper_device.hpp"

namespace ddj {

class GpuVecTest : public testing::Test
{
protected:
    GpuVecTest()
    {
        HelperDevice hc;
        hc.SetCudaDeviceWithMaxFreeMem();
    }

    ~GpuVecTest(){}
};

} /* namespace ddj */
#endif
