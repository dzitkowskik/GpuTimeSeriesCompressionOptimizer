#include "basic_thrust_histogram.hpp"

#include "helpers/helper_print.hpp"
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

namespace ddj {

SharedCudaPtrPair<int, int> BasicThrustHistogram::IntegerHistogram(SharedCudaPtr<int> data)
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
    thrust::upper_bound(
        thrust::device,
        input_keys_dvec.begin(),
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

//
// // sparse histogram using reduce_by_key
// template <typename Vector1,
//           typename Vector2,
//           typename Vector3>
// void sparse_histogram(const Vector1& input,
//                             Vector2& histogram_values,
//                             Vector3& histogram_counts)
// {
//   typedef typename Vector1::value_type ValueType; // input value type
//   typedef typename Vector3::value_type IndexType; // histogram index type
//
//   // copy input data (could be skipped if input is allowed to be modified)
//   thrust::device_vector<ValueType> data(input);
//
//   // print the initial data
//   print_vector("initial data", data);
//
//   // sort data to bring equal elements together
//   thrust::sort(data.begin(), data.end());
//
//   // print the sorted data
//   print_vector("sorted data", data);
//
//   // number of histogram bins is equal to number of unique values (assumes data.size() > 0)
//   IndexType num_bins = thrust::inner_product(data.begin(), data.end() - 1,
//                                              data.begin() + 1,
//                                              IndexType(1),
//                                              thrust::plus<IndexType>(),
//                                              thrust::not_equal_to<ValueType>());
//
//   // resize histogram storage
//   histogram_values.resize(num_bins);
//   histogram_counts.resize(num_bins);
//
//   // compact find the end of each bin of values
//   thrust::reduce_by_key(data.begin(), data.end(),
//                         thrust::constant_iterator<IndexType>(1),
//                         histogram_values.begin(),
//                         histogram_counts.begin());
//
//   // print the sparse histogram
//   print_vector("histogram values", histogram_values);
//   print_vector("histogram counts", histogram_counts);
// }

} /* namespace ddj */
