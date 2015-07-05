#include "basic_thrust_histogram.cuh"
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

} /* namespace ddj */
