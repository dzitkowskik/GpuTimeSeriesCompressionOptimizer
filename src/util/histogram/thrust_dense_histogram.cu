#include "thrust_dense_histogram.hpp"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/counting_iterator.h>

namespace ddj {

SharedCudaPtrPair<int, int> ThrustDenseHistogram::IntegerHistogram(SharedCudaPtr<int> data)
{
    thrust::device_ptr<int> data_ptr(data->get());
    thrust::device_vector<int> input_keys_dvec(data_ptr, data_ptr+data->size());

    thrust::sort(input_keys_dvec.begin(), input_keys_dvec.end());

    // number of histogram bins is equal to the maximum value plus one
    int num_bins = input_keys_dvec.back() - input_keys_dvec.front() + 1;

    // allocate histogram storage
    auto output_keys = CudaPtr<int>::make_shared(num_bins);
    auto output_counts = CudaPtr<int>::make_shared(num_bins);
    thrust::device_ptr<int> output_keys_ptr(output_keys->get());
    thrust::device_ptr<int> output_counts_ptr(output_counts->get());

    // find the end of each bin of values (cumulative histogram)
    thrust::counting_iterator<int> search_begin(input_keys_dvec.front());
    thrust::upper_bound(input_keys_dvec.begin(),
    					input_keys_dvec.end(),
						search_begin,
						search_begin + num_bins,
						output_counts_ptr);

    // compute the histogram by taking differences of the cumulative histogram
    thrust::adjacent_difference(output_counts_ptr, output_counts_ptr+num_bins, output_counts_ptr);

    // keys are sequence from input keys min to max
    thrust::counting_iterator<int> keys_begin(input_keys_dvec.front());
    thrust::copy(keys_begin, keys_begin+num_bins, output_keys_ptr);

    return SharedCudaPtrPair<int, int>(output_keys, output_counts);
}

} /* namespace ddj */
