/*
 * thrust_sparse_histogram.hpp
 *
 *  Created on: 14-08-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_UTIL_THRUST_SPARSE_HISTOGRAM_HPP_
#define DDJ_UTIL_THRUST_SPARSE_HISTOGRAM_HPP_

#include "core/cuda_ptr.hpp"
#include "histogram_base.hpp"

namespace ddj {

class ThrustSparseHistogram : public HistogramBase
{
public:
    virtual SharedCudaPtrPair<int, int> IntegerHistogram(SharedCudaPtr<int> data);
};

} /* namespace ddj */
#endif /* DDJ_UTIL_THRUST_SPARSE_HISTOGRAM_HPP_ */
