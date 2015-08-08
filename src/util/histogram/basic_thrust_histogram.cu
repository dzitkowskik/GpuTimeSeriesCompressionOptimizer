#include "basic_thrust_histogram.hpp"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

namespace ddj {

SharedCudaPtrPair<int, int> BasicThrustHistogram::IntegerHistogram(SharedCudaPtr<int> data)
{
    thrust::device_ptr<int> data_ptr(data->get());
    thrust::device_vector<int> output(data->size());
    thrust::device_vector<int> lengths(data->size());

    auto result = thrust::reduce_by_key(
        data_ptr,
        data_ptr+data->size(),
        thrust::constant_iterator<int>(1),
        output.begin(),
        lengths.begin());

    int len = result.first - output.begin();

    auto keys = CudaPtr<int>::make_shared(len);
    auto counts = CudaPtr<int>::make_shared(len);
    thrust::device_ptr<int> keys_ptr(keys->get());
    thrust::device_ptr<int> counts_ptr(counts->get());

    thrust::copy(output.begin(), output.begin()+len, keys_ptr);
    thrust::copy(lengths.begin(), lengths.begin()+len, counts_ptr);

    return SharedCudaPtrPair<int, int>(keys, counts);
}

// TODO: Chceck if dense has better performance
// dense histogram using binary search
// template <typename Vector1,
//           typename Vector2>
// void dense_histogram(const Vector1& input,
//                            Vector2& histogram)
// {
//   typedef typename Vector1::value_type ValueType; // input value type
//   typedef typename Vector2::value_type IndexType; // histogram index type
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
//   // number of histogram bins is equal to the maximum value plus one
//   IndexType num_bins = data.back() + 1;
//
//   // resize histogram storage
//   histogram.resize(num_bins);
//
//   // find the end of each bin of values
//   thrust::counting_iterator<IndexType> search_begin(0);
//   thrust::upper_bound(data.begin(), data.end(),
//                       search_begin, search_begin + num_bins,
//                       histogram.begin());
//
//   // print the cumulative histogram
//   print_vector("cumulative histogram", histogram);
//
//   // compute the histogram by taking differences of the cumulative histogram
//   thrust::adjacent_difference(histogram.begin(), histogram.end(),
//                               histogram.begin());
//
//   // print the histogram
//   print_vector("histogram", histogram);
// }

} /* namespace ddj */
