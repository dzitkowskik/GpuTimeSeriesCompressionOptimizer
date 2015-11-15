/*
 *  histogram.cpp
 *
 *  Created on: 7/10/2015
 *      Author: Karol Dzitkowski
 */

#include <util/histogram/histogram.hpp>
#include "util/statistics/cuda_array_statistics.hpp"


namespace ddj {

// INT
template<typename T>
SharedCudaPtr<int> Histogram::GetHistogram(SharedCudaPtr<T> data, int bucketCnt)
{
    return CudaHistogram(data, bucketCnt);
}

template<typename T>
SharedCudaPtrPair<T, int> Histogram::GetDictionaryCounter(SharedCudaPtr<T> data)
{
    return ThrustSparseHistogram(data);
}

// REST
template<typename T>
SharedCudaPtr<T> Histogram::GetMostFrequent(SharedCudaPtr<T> data, int freqCnt)
{
	auto histogram = GetDictionaryCounter(data);
	return GetMostFrequentSparse(histogram, freqCnt);
}

template<typename T>
SharedCudaPtr<T> Histogram::GetMostFrequent(SharedCudaPtrPair<T, int> histogram, int freqCnt)
{
	return GetMostFrequentSparse(histogram, freqCnt);
}

#define DELTA_ENCODING_SPEC(X) \
	template SharedCudaPtr<X> Histogram::GetMostFrequent<X>(SharedCudaPtr<X>, int); \
	template SharedCudaPtr<X> Histogram::GetMostFrequent<X>(SharedCudaPtrPair<X, int>, int); \
	template SharedCudaPtr<int> Histogram::GetHistogram<X>(SharedCudaPtr<X>, int); \
	template SharedCudaPtrPair<X, int> Histogram::GetDictionaryCounter<X>(SharedCudaPtr<X>);
FOR_EACH(DELTA_ENCODING_SPEC, float, int, long, long long, unsigned int)


} /* namespace ddj */
