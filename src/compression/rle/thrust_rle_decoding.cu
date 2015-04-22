#include "thrust_rle.cuh"
#include "../../helpers/helper_print.h"

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>
#include <iterator>

namespace ddj {

template<typename T>
T* ThrustRleCompression::Decode(void* data, int in_size, int& out_size)
{
    // prepare input data
    int* lengths_data = reinterpret_cast<int*>(data);
    T* values_data = reinterpret_cast<T*>(lengths_data + in_size);

    thrust::device_ptr<int> dev_ptr_lengths(lengths_data);
    thrust::device_ptr<T> dev_ptr_float(values_data);

    thrust::device_vector<T> input(dev_ptr_float, dev_ptr_float + in_size);
    thrust::device_vector<int> lengths(dev_ptr_lengths, dev_ptr_lengths + in_size);

    // scan the lengths
    thrust::inclusive_scan(lengths.begin(), lengths.end(), lengths.begin());

    // output size is sum of the run lengths
    int N = lengths.back();

    // compute input index for each output element
    thrust::device_vector<int> indices(N);
    thrust::lower_bound(
        lengths.begin(),
        lengths.end(),
        thrust::counting_iterator<int>(1),
        thrust::counting_iterator<int>(N + 1),
        indices.begin());

    // gather input elements

    T* raw_ptr;
    cudaMalloc((void **) &raw_ptr, N * sizeof(T));
    thrust::device_ptr<T> dev_ptr(raw_ptr);

    thrust::gather(indices.begin(), indices.end(), input.begin(), dev_ptr);

    #if DDJ_THRUST_RLE_DEBUG
    HelperPrint::PrintDeviceVectors(input, lengths, "Thrust RLE decoding input");
    HelperPrint::PrintDevicePtr(dev_ptr, N, "Thrust RLE decoding output");
    #endif

    out_size = N;
    return raw_ptr;
}

template float* ThrustRleCompression::Decode<float>(void* data, int in_size, int& out_size);

} /* namespace ddj */
