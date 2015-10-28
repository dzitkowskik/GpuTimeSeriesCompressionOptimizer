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
	template<typename T> SharedCudaPtrPair<T, int> CalculateSparse(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtrPair<T, int> CalculateDense(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtrPair<T, int> CalculateBuckets(SharedCudaPtr<T> data, int bucketCnt);
	template<typename T> SharedCudaPtr<T> GetMostFrequent(SharedCudaPtr<T> data, int freqCnt);
	template<typename T> SharedCudaPtr<T> GetMostFrequent(SharedCudaPtrPair<T, int> histogram, int freqCnt);

private:
	template<typename T> SharedCudaPtrPair<T, int> ThrustSparseHistogram(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtrPair<T, int> ThrustDenseHistogram(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtrPair<T, int> CudaHistogramIntegral(SharedCudaPtr<T> data, T min, T max);
	template<typename T> SharedCudaPtrPair<T, int> CudaHistogramBuckets(SharedCudaPtr<T> data, T min, T max);

	template<typename T> SharedCudaPtr<T> GetMostFrequentSparse(SharedCudaPtrPair<T, int>, int);

private:
	CudaArrayTransform _transform;

	friend class HistogramTest;
 	FRIEND_TEST(HistogramTest, ThrustSparseHistogram_RandomIntegerArray);
	FRIEND_TEST(HistogramTest, ThrustSparseHistogram_RealData_Delta_Time);
 	FRIEND_TEST(HistogramTest, GetMostFrequent_fake_data);
	FRIEND_TEST(HistogramTest, GetMostFrequent_random_int);

};

} /* namespace ddj */

#endif /* DDJ_HISTOGRAM_HPP_ */
