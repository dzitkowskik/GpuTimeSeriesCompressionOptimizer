/*
 * cuda_array_reduce.cuh
 *
 *  Created on: 3 lis 2015
 *      Author: ghash
 */

#ifndef CUDA_ARRAY_REDUCE_CUH_
#define CUDA_ARRAY_REDUCE_CUH_

#include "core/cuda_ptr.hpp"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

template<typename T, typename Predicate> inline
T reduce_thrust(SharedCudaPtr<T> data, Predicate pred)
{
	T result = 0;
    if(data->get() != NULL && data->size() != 0)
    {
    	thrust::device_ptr<T> data_ptr(data->get());
    	result = thrust::reduce(data_ptr, data_ptr+data->size(), (T)0, pred);
    }
    return result;
}


#endif /* CUDA_ARRAY_REDUCE_CUH_ */
