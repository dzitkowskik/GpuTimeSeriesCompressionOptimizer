#include "util/histogram/histogram.hpp"
#include "util/histogram/cuda_histogram_impl.cuh"
#include "util/generator/cuda_array_generator.hpp"
#include "util/statistics/cuda_array_statistics.hpp"
#include "core/macros.h"
#include <cmath>

namespace ddj {

template<typename T>
struct HistForm
{
	T min;
	double invStep;

    __host__ __device__
    void operator() (T* input, int i, int* res_idx, int* res, int nres) const
    {
        *res_idx++ = (input[i] - min) * invStep;
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
SharedCudaPtr<int> Histogram::CudaHistogram(SharedCudaPtr<T> data, int bucketCnt)
{
	auto minMax = CudaArrayStatistics().MinMax(data);
	T min = std::get<0>(minMax);
	T max = std::get<1>(minMax);

    auto distance = max - min;
    auto counts = CudaPtr<int>::make_shared(bucketCnt);
    this->_transform.TransformInPlace(counts, ZeroOperator<int, int>());
    auto invStep = (double)(bucketCnt-1) / distance;

    HistForm<T> xform { min, invStep };
    HistSumFun sum;

    callHistogramKernel<histogram_generic, 1>(
    		data->get(),
    		xform,
    		sum,
    		0,
    		(int)data->size(),
    		0,
    		counts->get(),
    		bucketCnt,
    		true);

    return counts;
}

#define CUDA_HISTOGRAM_SPEC(X) \
	template SharedCudaPtr<int> Histogram::CudaHistogram<X>(SharedCudaPtr<X>, int);
FOR_EACH(CUDA_HISTOGRAM_SPEC, char, short, double, float, int, long, long long, unsigned int)

} /* namespace ddj */
