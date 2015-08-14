/*
 * thrust_dense_histogram.cuh 04-07-2015 Karol Dzitkowski
 */

#ifndef DDJ_UTIL_THRUST_DENSE_HISTOGRAM_CUH_
#define DDJ_UTIL_THRUST_DENSE_HISTOGRAM_CUH_

#include "core/cuda_ptr.hpp"
#include "histogram_base.hpp"

namespace ddj {

class ThrustDenseHistogram : public HistogramBase
{
public:
    virtual SharedCudaPtrPair<int, int> IntegerHistogram(SharedCudaPtr<int> data);
};

} /* namespace ddj */
#endif /* DDJ_UTIL_THRUST_DENSE_HISTOGRAM_CUH_ */
