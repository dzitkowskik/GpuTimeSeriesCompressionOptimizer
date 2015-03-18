#ifndef DDJ_GPUVEC_UNITTEST_H_
#define DDJ_GPUVEC_UNITTEST_H_

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../gpuvec.h"
#include "../cudacommons.h"
#include "../helper_cuda.h"

namespace ddj {

class GpuVecTest : public testing::Test
{
protected:
    GpuVecTest()
    {
        CudaCommons cudaC;
        cudaC.SetCudaDeviceWithMaxFreeMem();
    }

    ~GpuVecTest(){}

    virtual void SetUp()
    {
        const char* argv = "";
        cudaSetDevice(findCudaDevice(0, &argv));
    }
};

} /* namespace ddj */
#endif

