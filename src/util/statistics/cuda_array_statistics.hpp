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

class CudaArrayStatistics
{
public:
    template<typename T> std::tuple<T,T> MinMax(SharedCudaPtr<T> data);

private:
    ExecutionPolicy _policy;
};

} /* namespace ddj */
#endif /* DDJ_UTIL_CUDA_ARRAY_STATISTICS_HPP_ */
