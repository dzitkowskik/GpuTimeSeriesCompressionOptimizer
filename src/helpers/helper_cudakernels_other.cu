/*
 * helper_cudakernels_other.cu
 *
 *  Created on: 09-04-2015
 *      Author: Karol Dzitkowski
 */

#include "helper_cudakernels.cuh"
#include "core/operators.cuh"
#include "core/execution_policy.hpp"
#include "helpers/helper_cuda.cuh"
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>

#define MODULO_KERNEL_BLOCK_SIZE 32
#define CREATE_NUM_KERNEL_BLOCK_SIZE 32

namespace ddj
{

template<typename T> std::tuple<T,T> HelperCudaKernels::MinMax(SharedCudaPtr<T> data)
{
	thrust::device_ptr<T> dp(data->get());
	auto tuple = thrust::minmax_element(dp, dp+data->size());
	T min = *(tuple.first);
	T max = *(tuple.second);
	return std::make_tuple(min, max);
}

#define CUDA_KERNELS_OTHER_SPEC(X) \
	template std::tuple<X,X> HelperCudaKernels::MinMax<X>(SharedCudaPtr<X> data);
FOR_EACH(CUDA_KERNELS_OTHER_SPEC, float, int, long long, unsigned int)

} /* namespace ddj */
