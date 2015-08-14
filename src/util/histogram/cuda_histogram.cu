#include "util/histogram/cuda_histogram.hpp"
#include "util/histogram/cuda_histogram_impl.cuh"
#include <cmath>

namespace ddj {

struct test_xform
{
    __host__ __device__
    void operator() (int* input, int i, int* res_idx, int* res, int nres) const
    {
        *res_idx++ = input[i];
        *res++ = 1;
    }
};

// Sum-functor to be used for reduction - just a normal sum of two integers
struct test_sumfun
{
    __device__ __host__
    int operator() (int res1, int res2) const
    {
        return res1 + res2;
    }
};

SharedCudaPtrPair<int, int> CudaHistogram::IntegerHistogram(SharedCudaPtr<int> data)
{
    auto minMax = this->_cudaKernels.MinMax(data);
    int distance = std::abs(std::get<1>(minMax) - std::get<0>(minMax)) + 1;
    auto counts = CudaPtr<int>::make_shared(distance);
    test_xform xform;
    test_sumfun sum;
    callHistogramKernel<histogram_atomic_inc, 1>
      (data->get(), xform, sum, 0, (int)data->size(), 0, counts->get(), distance);
    auto keys = this->_cudaKernels.CreateConsecutiveNumbersArray(distance, std::get<0>(minMax));
    return SharedCudaPtrPair<int, int>(keys, counts);
}

} /* namespace ddj */
