#include "util/histogram/histogram.hpp"
#include "core/macros.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/counting_iterator.h>

namespace ddj {

template<typename T>
SharedCudaPtrPair<T, int> Histogram::ThrustDenseHistogram(SharedCudaPtr<T> data)
{
    thrust::device_ptr<T> data_ptr(data->get());
    thrust::device_vector<T> input_keys_dvec(data_ptr, data_ptr+data->size());

    thrust::sort(input_keys_dvec.begin(), input_keys_dvec.end());

    // number of histogram bins is equal to the maximum value plus one
    int num_bins = input_keys_dvec.back() - input_keys_dvec.front() + 1;

    // allocate histogram storage
    auto output_keys = CudaPtr<T>::make_shared(num_bins);
    auto output_counts = CudaPtr<int>::make_shared(num_bins);
    thrust::device_ptr<T> output_keys_ptr(output_keys->get());
    thrust::device_ptr<int> output_counts_ptr(output_counts->get());
    thrust::device_vector<int> output_counts_vec(num_bins);

    // find the end of each bin of values (cumulative histogram)
    thrust::counting_iterator<T> search_begin(input_keys_dvec.front());
    thrust::upper_bound(input_keys_dvec.begin(),
    					input_keys_dvec.end(),
						search_begin,
						search_begin + num_bins,
						output_counts_vec.begin());

    // compute the histogram by taking differences of the cumulative histogram
    thrust::adjacent_difference(output_counts_vec.begin(), output_counts_vec.end(), output_counts_vec.begin());

    // keys are sequence from input keys min to max
    thrust::counting_iterator<T> keys_begin(input_keys_dvec.front());
    thrust::copy(keys_begin, keys_begin+num_bins, output_keys_ptr);
    thrust::copy(output_counts_vec.begin(), output_counts_vec.end(), output_counts_ptr);

    return SharedCudaPtrPair<T, int>(output_keys, output_counts);
}

#define THRUST_DENSE_HISTOGRAM_SPEC(X) \
	template SharedCudaPtrPair<X, int> Histogram::ThrustDenseHistogram<X>(SharedCudaPtr<X>);
FOR_EACH(THRUST_DENSE_HISTOGRAM_SPEC, char, short, double, float, int, long, long long, unsigned int)

} /* namespace ddj */
