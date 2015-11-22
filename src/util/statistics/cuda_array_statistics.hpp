/*
 *  cuda_array_statistics.hpp
 *
 *  Created on: 21-10-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_UTIL_CUDA_ARRAY_STATISTICS_HPP_
#define DDJ_UTIL_CUDA_ARRAY_STATISTICS_HPP_

#include "core/cuda_ptr.hpp"
#include "core/execution_policy.hpp"

namespace ddj
{

__host__ __device__ int _getFloatPrecision(float number);

template<typename T>
struct Statistics
{
	T min;
	T max;
	char minBitCnt;
	int precision;
	bool sorted;
	float rlMetric;
	T mean;
};

class CudaArrayStatistics
{
public:
    template<typename T> std::tuple<T,T> MinMax(SharedCudaPtr<T> data);
    template<typename T> char MinBitCnt(SharedCudaPtr<T> data);
    template<typename T> int Precision(SharedCudaPtr<T> data);
    template<typename T> bool Sorted(SharedCudaPtr<T> data);
    template<typename T, int N=3> float RlMetric(SharedCudaPtr<T> data);
    template<typename T> T Mean(SharedCudaPtr<T> data);

    template<typename T> Statistics<T> GenerateStatistics(SharedCudaPtr<T> data);

private:
    ExecutionPolicy _policy;
};

} /* namespace ddj */
#endif /* DDJ_UTIL_CUDA_ARRAY_STATISTICS_HPP_ */
