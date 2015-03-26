#ifndef DDJ_GPUVEC_UNITTEST_H_
#define DDJ_GPUVEC_UNITTEST_H_

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../store/gpuvec.h"
#include "../helpers/helper_cuda.h"

namespace ddj {

class GpuVecTest : public testing::Test
{
protected:
    GpuVecTest()
    {
        HelperCuda hc;
        hc.SetCudaDeviceWithMaxFreeMem();
    }

    ~GpuVecTest(){}

    virtual void SetUp()
    {
    }
};

} /* namespace ddj */
#endif

