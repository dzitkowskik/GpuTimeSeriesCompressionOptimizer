/*
 * histogram_base.hpp 04-07-2015 Karol Dzitkowski
 */

#ifndef DDJ_UTIL_HISTOGRAM_BASE_CUH_
#define DDJ_UTIL_HISTOGRAM_BASE_CUH_

namespace ddj {

class HistogramBase
{
public:
    virtual SharedCudaPtr<int> IntegerHistogram(SharedCudaPtr<int> data) = 0;
};

} /* namespace ddj */
#endif /* DDJ_UTIL_HISTOGRAM_BASE_CUH_ */
