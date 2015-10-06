/*
 * histogram_base.hpp 04-07-2015 Karol Dzitkowski
 */

#ifndef DDJ_UTIL_HISTOGRAM_BASE_CUH_
#define DDJ_UTIL_HISTOGRAM_BASE_CUH_

#include "core/cuda_ptr.hpp"

namespace ddj {

class HistogramBase
{
public:
	HistogramBase(){}
	virtual ~HistogramBase(){}
	HistogramBase(const HistogramBase&) = default;
	HistogramBase(HistogramBase&&) = default;

public:
    virtual SharedCudaPtrPair<int, int> IntegerHistogram(SharedCudaPtr<int> data) = 0;
};

// TODO: Implement other algorithms of calculating histogram on GPU
// http://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/

} /* namespace ddj */
#endif /* DDJ_UTIL_HISTOGRAM_BASE_CUH_ */
