#include "util/histogram/cuda_histogram.hpp"
#include "util/histogram/cuda_histogram_impl.cuh"
#include "core/operators.cuh"
#include <cmath>

namespace ddj {

struct test_xform
{
	int min;

    __host__ __device__
    void operator() (int* input, int i, int* res_idx, int* res, int nres) const
    {
        *res_idx++ = input[i] - min;
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
    this->_transform.TransformInPlace(counts, ZeroOperator<int>());

    test_xform xform { std::get<0>(minMax) };
    test_sumfun sum;

    callHistogramKernel<histogram_atomic_add, 1>
      (data->get(), xform, sum, 0, (int)data->size(), 0, counts->get(), distance, true);

    auto keys = this->_cudaKernels.CreateConsecutiveNumbersArray(distance, std::get<0>(minMax));
    return SharedCudaPtrPair<int, int>(keys, counts);
}

} /* namespace ddj */
