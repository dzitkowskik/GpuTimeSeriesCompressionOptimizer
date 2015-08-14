/*
 * thrust_sparse_histogram.cpp
 *
 *  Created on: 14-08-2015
 *      Author: Karol Dzitkowski
 */

#include "thrust_sparse_histogram.hpp"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>

namespace ddj {


SharedCudaPtrPair<int, int> ThrustSparseHistogram::IntegerHistogram(SharedCudaPtr<int> data)
{
	thrust::device_ptr<int> data_ptr(data->get());
	thrust::device_vector<int> input_keys_dvec(data_ptr, data_ptr+data->size());

	if(data->size() <= 0)
		return SharedCudaPtrPair<int, int>(CudaPtr<int>::make_shared(), CudaPtr<int>::make_shared());

	thrust::sort(input_keys_dvec.begin(), input_keys_dvec.end());

	// number of histogram bins is equal to number of unique values (assumes data.size() > 0)
	int num_bins = thrust::inner_product( input_keys_dvec.begin(),
												input_keys_dvec.end() - 1,
												input_keys_dvec.begin() + 1,
												int(1),
												thrust::plus<int>(),
												thrust::not_equal_to<int>());

	// allocate histogram storage
	auto output_keys = CudaPtr<int>::make_shared(num_bins);
	auto output_counts = CudaPtr<int>::make_shared(num_bins);
	thrust::device_ptr<int> output_keys_ptr(output_keys->get());
	thrust::device_ptr<int> output_counts_ptr(output_counts->get());

	// compact find the end of each bin of values
	thrust::reduce_by_key(  input_keys_dvec.begin(),
							input_keys_dvec.end(),
							thrust::constant_iterator<int>(1),
							output_keys_ptr,
							output_counts_ptr);

	return SharedCudaPtrPair<int, int>(output_keys, output_counts);
}

} /* namespace ddj */
