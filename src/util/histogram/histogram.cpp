/*
 *  histogram.cpp
 *
 *  Created on: 7/10/2015
 *      Author: Karol Dzitkowski
 */

#include <util/histogram/histogram.hpp>

namespace ddj {

// INT
template<> SharedCudaPtrPair<int, int> Histogram::Calculate<int>(SharedCudaPtr<int> data)
{
	return CudaHistogramIntegral<int>(data);
}

// FLOAT
template<> SharedCudaPtrPair<float, int> Histogram::Calculate<float>(SharedCudaPtr<float> data)
{
	return ThrustSparseHistogram<float>(data);
}

} /* namespace ddj */
