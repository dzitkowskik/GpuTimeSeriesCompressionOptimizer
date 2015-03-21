#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include "thrust_rle.cuh"

#include <iostream>
#include <iterator>

namespace ddj {

void* ThrustRleCompression::Decode(void* data, int in_size, int& out_size)
{
    // prepare input data
    int* lengths_data = reinterpret_cast<int*>(data);
    float* values_data = reinterpret_cast<float*>(lengths_data + in_size);

    thrust::device_ptr<int> dev_ptr_lengths(lengths_data);
    thrust::device_ptr<float> dev_ptr_float(values_data);

    thrust::device_vector<float> input(dev_ptr_float, dev_ptr_float + in_size);
    thrust::device_vector<int> lengths(dev_ptr_lengths, dev_ptr_lengths + in_size);

    #if DDJ_THRUST_RLE_DEBUG
        // print the initial data
        std::cout << "run-length encoded input:" << std::endl;
        for(size_t i = 0; i < in_size; i++)
            std::cout << "(" << input[i] << "," << lengths[i] << ")";
        std::cout << std::endl << std::endl;
    #endif

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

    float* raw_ptr;
    cudaMalloc((void **) &raw_ptr, N * sizeof(float));
    thrust::device_ptr<float> dev_ptr(raw_ptr);

    thrust::gather(indices.begin(), indices.end(), input.begin(), dev_ptr);

    #if DDJ_THRUST_RLE_DEBUG
        // print the initial data
        std::cout << "decoded output:" << std::endl;
        thrust::copy(dev_ptr, dev_ptr + N, std::ostream_iterator<float>(std::cout, ""));
        std::cout << std::endl;
    #endif

    out_size = N;
    return raw_ptr;
}

} /* namespace ddj */
