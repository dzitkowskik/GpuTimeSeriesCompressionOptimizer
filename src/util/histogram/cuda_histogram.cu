#include "util/histogram/histogram.hpp"
#include "util/histogram/cuda_histogram_impl.cuh"
#include "core/operators.cuh"
#include "util/generator/cuda_array_generator.hpp"
#include "core/macros.h"
#include <cmath>

namespace ddj {

template<typename T>
struct HistIntegralForm
{
	T min;

    __host__ __device__
    void operator() (T* input, int i, int* res_idx, int* res, int nres) const
    {
        *res_idx++ = (int)(input[i] - min);
        *res++ = 1;
    }
};

struct HistIntegralSumFun
{
    __device__ __host__
    int operator() (int res1, int res2) const
    { return res1 + res2; }
};

template<typename T>
SharedCudaPtrPair<T, int> Histogram::CudaHistogramIntegral(SharedCudaPtr<T> data, T min, T max)
{
	static_assert(std::is_integral<T>::value, "CudaHistogramIntegral allows only integral types");
    int distance = max - min + 1;
    auto counts = CudaPtr<int>::make_shared(distance);
    this->_transform.TransformInPlace(counts, ZeroOperator<int>());

    HistIntegralForm<T> xform { min };
    HistIntegralSumFun sum;

    callHistogramKernel<histogram_atomic_inc, 1>
      (data->get(), xform, sum, 0, (int)data->size(), 0, counts->get(), distance, true);

    auto keys = CudaArrayGenerator().CreateConsecutiveNumbersArray<T>(distance, min);
    return SharedCudaPtrPair<T, int>(keys, counts);
}

template<typename T>
struct HistForm
{
	T min;
	T invStep;

    __host__ __device__
    void operator() (T* input, int i, int* res_idx, int* res, int nres) const
    {
        *res_idx++ = (int)(input[i] - min) * invStep;
        *res++ = 1;
    }
};

struct HistSumFun
{
    __device__ __host__
    int operator() (int res1, int res2) const
    { return res1 + res2; }
};

template<typename T>
SharedCudaPtrPair<T, int> Histogram::CudaHistogramBuckets(SharedCudaPtr<T> data, T min, T max)
{
    auto distance = max - min;
    auto counts = CudaPtr<int>::make_shared(data->size());
    this->_transform.TransformInPlace(counts, ZeroOperator<int>());
    T step = distance / data->size();
    T invStep = data->size() / distance;

    HistForm<T> xform { min, invStep };
    HistSumFun sum;

    callHistogramKernel<histogram_generic, 1>
      (data->get(), xform, sum, 0, (int)data->size(), 0, counts->get(), distance, true);

    auto keys = CudaArrayGenerator().CreateConsecutiveNumbersArray<T>(distance, min, step);
    return SharedCudaPtrPair<T, int>(keys, counts);
}

#define CUDA_HISTOGRAM_INTEGRAL_SPEC(X) \
	template SharedCudaPtrPair<X, int> Histogram::CudaHistogramIntegral<X>(SharedCudaPtr<X>, X, X); \
	template SharedCudaPtrPair<X, int> Histogram::CudaHistogramBuckets<X>(SharedCudaPtr<X>, X, X);
FOR_EACH(CUDA_HISTOGRAM_INTEGRAL_SPEC, int, long, long long, unsigned int)

#define CUDA_HISTOGRAM_FLOATING_POINT_SPEC(X) \
	template SharedCudaPtrPair<X, int> Histogram::CudaHistogramBuckets<X>(SharedCudaPtr<X>, X, X);
FOR_EACH(CUDA_HISTOGRAM_FLOATING_POINT_SPEC, float)

} /* namespace ddj */
