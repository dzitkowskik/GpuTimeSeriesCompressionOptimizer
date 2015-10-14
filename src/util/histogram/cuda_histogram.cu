#include "util/histogram/histogram.hpp"
#include "util/histogram/cuda_histogram_impl.cuh"
#include "core/operators.cuh"
#include "util/generator/cuda_array_generator.hpp"
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
SharedCudaPtrPair<T, int> Histogram::CudaHistogramIntegral(SharedCudaPtr<T> data)
{
	static_assert(std::is_integral<T>::value, "CudaHistogramIntegral allows only integral types");

    auto minMax = this->_cudaKernels.MinMax<T>(data);
    int distance = std::get<1>(minMax) - std::get<0>(minMax) + 1;

    auto counts = CudaPtr<int>::make_shared(distance);
    this->_transform.TransformInPlace(counts, ZeroOperator<int>());

    HistIntegralForm<T> xform { std::get<0>(minMax) };
    HistIntegralSumFun sum;

    callHistogramKernel<histogram_atomic_add, 1>
      (data->get(), xform, sum, 0, (int)data->size(), 0, counts->get(), distance, true);

    auto keys = CudaArrayGenerator().CreateConsecutiveNumbersArray<T>(distance, std::get<0>(minMax));
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
SharedCudaPtrPair<T, int> Histogram::CudaHistogram(SharedCudaPtr<T> data)
{
    auto minMax = this->_cudaKernels.MinMax<T>(data);
    auto distance = std::get<1>(minMax) - std::get<0>(minMax);

    auto counts = CudaPtr<int>::make_shared(data->size());
    this->_transform.TransformInPlace(counts, ZeroOperator<int>());
    T step = distance / data->size();
    T invStep = data->size() / distance;

    HistForm<T> xform { std::get<0>(minMax), invStep };
    HistSumFun sum;

    callHistogramKernel<histogram_generic, 1>
      (data->get(), xform, sum, 0, (int)data->size(), 0, counts->get(), distance, true);

    auto keys = CudaArrayGenerator().CreateConsecutiveNumbersArray<T>(distance, std::get<0>(minMax), step);
    return SharedCudaPtrPair<T, int>(keys, counts);
}

#define CUDA_HISTOGRAM_INTEGRAL_SPEC(X) \
	template SharedCudaPtrPair<X, int> Histogram::CudaHistogramIntegral<X>(SharedCudaPtr<X>); \
	template SharedCudaPtrPair<X, int> Histogram::CudaHistogram<X>(SharedCudaPtr<X>);
FOR_EACH(CUDA_HISTOGRAM_INTEGRAL_SPEC, int, long long, unsigned int)

#define CUDA_HISTOGRAM_FLOATING_POINT_SPEC(X) \
	template SharedCudaPtrPair<X, int> Histogram::CudaHistogram<X>(SharedCudaPtr<X>);
FOR_EACH(CUDA_HISTOGRAM_FLOATING_POINT_SPEC, float)

} /* namespace ddj */
