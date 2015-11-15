/*
 *  histogram.hpp
 *
 *  Created on: 7/10/2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_HISTOGRAM_HPP_
#define DDJ_HISTOGRAM_HPP_

#include "util/transform/cuda_array_transform.hpp"
#include <gtest/gtest.h>

namespace ddj {

class Histogram
{
public:
	template<typename T> SharedCudaPtrPair<T, int> GetDictionaryCounter(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<int> GetHistogram(SharedCudaPtr<T> data, int bucketCnt);
	template<typename T> SharedCudaPtr<T> GetMostFrequent(SharedCudaPtr<T> data, int freqCnt);
	template<typename T> SharedCudaPtr<T> GetMostFrequent(SharedCudaPtrPair<T, int> histogram, int freqCnt);

public:
	template<typename T> __host__ __device__ int GetBucket(T value, T min, T max, int bucketCnt)
	{
		auto distance = max - min;
		T invStep = bucketCnt / distance;
		return (int)(value - min) * invStep;
	}

private:
	template<typename T> SharedCudaPtrPair<T, int> ThrustSparseHistogram(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtrPair<T, int> ThrustDenseHistogram(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<int> CudaHistogram(SharedCudaPtr<T> data, int bucketCnt);
	template<typename T> SharedCudaPtr<T> GetMostFrequentSparse(SharedCudaPtrPair<T, int>, int);

private:
	CudaArrayTransform _transform;
};

} /* namespace ddj */

#endif /* DDJ_HISTOGRAM_HPP_ */
