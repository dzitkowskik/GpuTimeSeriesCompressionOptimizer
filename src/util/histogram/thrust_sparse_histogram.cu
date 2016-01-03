/*
 * thrust_sparse_histogram.cpp
 *
 *  Created on: 14-08-2015
 *      Author: Karol Dzitkowski
 */

#include "util/histogram/histogram.hpp"
#include "core/macros.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>

namespace ddj {

template<typename T>
SharedCudaPtrPair<T, int> Histogram::ThrustSparseHistogram(SharedCudaPtr<T> data)
{
	thrust::device_ptr<T> data_ptr(data->get());
	thrust::device_vector<T> input_keys_dvec(data_ptr, data_ptr+data->size());

	if(data->size() <= 0)
		return SharedCudaPtrPair<T, int>(CudaPtr<T>::make_shared(), CudaPtr<int>::make_shared());

	thrust::sort(input_keys_dvec.begin(), input_keys_dvec.end());

	// number of histogram bins is equal to number of unique values (assumes data.size() > 0)
	int num_bins = thrust::inner_product( 	input_keys_dvec.begin(),
											input_keys_dvec.end() - 1,
											input_keys_dvec.begin() + 1,
											int(1),
											thrust::plus<int>(),
											thrust::not_equal_to<T>());

	// allocate histogram storage
	auto output_keys = CudaPtr<T>::make_shared(num_bins);
	auto output_counts = CudaPtr<int>::make_shared(num_bins);
	thrust::device_ptr<T> output_keys_ptr(output_keys->get());
	thrust::device_ptr<int> output_counts_ptr(output_counts->get());

	// compact find the end of each bin of values
	thrust::reduce_by_key(  input_keys_dvec.begin(),
							input_keys_dvec.end(),
							thrust::constant_iterator<int>(1),
							output_keys_ptr,
							output_counts_ptr);

	return SharedCudaPtrPair<T, int>(output_keys, output_counts);
}

#define THRUST_SPARSE_HISTOGRAM_SPEC(X) \
	template SharedCudaPtrPair<X, int> Histogram::ThrustSparseHistogram<X>(SharedCudaPtr<X>);
FOR_EACH(THRUST_SPARSE_HISTOGRAM_SPEC, char, short, double, float, int, long, long long, unsigned int)

} /* namespace ddj */
