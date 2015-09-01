/*
 * cuda_histogram.cuh 06-08-2015 Karol Dzitkowski
 */

#ifndef DDJ_UTIL_CUDA_HISTOGRAM_HPP_
#define DDJ_UTIL_CUDA_HISTOGRAM_HPP_

#include "core/cuda_ptr.hpp"
#include "histogram_base.hpp"
#include "helpers/helper_cudakernels.cuh"

namespace ddj {

class CudaHistogram : public HistogramBase
{

public:
    virtual SharedCudaPtrPair<int, int> IntegerHistogram(SharedCudaPtr<int> data);

    // TODO: Implement Histogram also as a template

private:
    HelperCudaKernels _cudaKernels;

};

} /* namespace ddj */
#endif /* DDJ_UTIL_CUDA_HISTOGRAM_HPP_ */
