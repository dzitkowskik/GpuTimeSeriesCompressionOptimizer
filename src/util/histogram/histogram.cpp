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
	return CudaHistogramIntegral(data);
}

template<> SharedCudaPtr<int> Histogram::GetMostFrequent(SharedCudaPtr<int> data, int freqCnt)
{
	auto histogram = ThrustSparseHistogram(data);
	return GetMostFrequentSparse(histogram, freqCnt);
}

template<> SharedCudaPtr<int> Histogram::GetMostFrequent(
		SharedCudaPtrPair<int, int> histogram, int freqCnt)
{
	return GetMostFrequentSparse(histogram, freqCnt);
}

// FLOAT
template<> SharedCudaPtrPair<float, int> Histogram::Calculate<float>(SharedCudaPtr<float> data)
{
	return ThrustSparseHistogram(data);
}

template<> SharedCudaPtr<float> Histogram::GetMostFrequent(SharedCudaPtr<float> data, int freqCnt)
{
	auto histogram = ThrustSparseHistogram(data);
	return GetMostFrequentSparse(histogram, freqCnt);
}

template<> SharedCudaPtr<float> Histogram::GetMostFrequent(
		SharedCudaPtrPair<float, int> histogram, int freqCnt)
{
	return GetMostFrequentSparse(histogram, freqCnt);
}

} /* namespace ddj */
