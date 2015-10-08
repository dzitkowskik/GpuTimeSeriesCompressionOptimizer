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

template<> SharedCudaPtr<int> Histogram::GetMostFrequent(SharedCudaPtrPair<int, int> histogram, int freqCnt)
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

// REST
template<typename T>
SharedCudaPtr<T> Histogram::GetMostFrequent(SharedCudaPtr<T> data, int freqCnt)
{
	auto histogram = ThrustSparseHistogram(data);
	return GetMostFrequentSparse(histogram, freqCnt);
}

template<typename T>
SharedCudaPtr<T> Histogram::GetMostFrequent(SharedCudaPtrPair<T, int> histogram, int freqCnt)
{
	return GetMostFrequentSparse(histogram, freqCnt);
}

#define DELTA_ENCODING_SPEC(X) \
	template SharedCudaPtr<X> Histogram::GetMostFrequent<X>(SharedCudaPtr<X>, int); \
	template SharedCudaPtr<X> Histogram::GetMostFrequent<X>(SharedCudaPtrPair<X, int>, int);
FOR_EACH(DELTA_ENCODING_SPEC, long long, unsigned int)


} /* namespace ddj */
