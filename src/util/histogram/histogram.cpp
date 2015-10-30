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
template<> SharedCudaPtrPair<int, int> Histogram::CalculateDense<int>(SharedCudaPtr<int> data)
{
    auto minMax = CudaArrayStatistics().MinMax<int>(data);
    if(std::get<1>(minMax) - std::get<0>(minMax) < 1000)
    	return CudaHistogramIntegral(data, std::get<0>(minMax), std::get<1>(minMax));
    else
    	return ThrustDenseHistogram(data);
}

template<> SharedCudaPtrPair<int, int> Histogram::CalculateSparse<int>(SharedCudaPtr<int> data)
{
    return ThrustSparseHistogram(data);
}

// LONG
template<> SharedCudaPtrPair<long, int> Histogram::CalculateDense<long>(SharedCudaPtr<long> data)
{
    auto minMax = CudaArrayStatistics().MinMax<long>(data);
    if(std::get<1>(minMax) - std::get<0>(minMax) < 1000)
    	return CudaHistogramIntegral(data, std::get<0>(minMax), std::get<1>(minMax));
    else
    	return ThrustDenseHistogram(data);
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
FOR_EACH(DELTA_ENCODING_SPEC, float, int, long, long long, unsigned int)


} /* namespace ddj */
