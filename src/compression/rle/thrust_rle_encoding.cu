#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include "thrust_rle.cuh"

#include <iostream>
#include <iterator>

namespace ddj {

void* ThrustRleCompression::Encode(void* data, const int in_size, int& out_size)
{
    thrust::device_ptr<float> d_ptr(reinterpret_cast<float*>(data));
    thrust::device_vector<float> input(d_ptr, d_ptr + in_size);
    thrust::device_vector<float> output(in_size);
    thrust::device_vector<int>  lengths(in_size);

    #if DDJ_THRUST_RLE_DEBUG
        // print the initial data
        std::cout << "input data:" << std::endl;
        thrust::copy(input.begin(), input.end(), std::ostream_iterator<float>(std::cout, ""));
        std::cout << std::endl << std::endl;
    #endif

    // compute run lengths
    auto result = thrust::reduce_by_key(
        input.begin(),
        input.end(),
        thrust::constant_iterator<int>(1),
        output.begin(),
        lengths.begin());

    size_t len = result.first - output.begin();

    #if DDJ_THRUST_RLE_DEBUG
        // print the output
        std::cout << "run-length encoded output:" << std::endl;
        for(size_t i = 0; i < len; i++)
            std::cout << "(" << output[i] << "," << lengths[i] << ")";
        std::cout << std::endl;
    #endif

    // prepare data
    int* raw_ptr;
    cudaMalloc((void **) &raw_ptr, len * sizeof(int) + len * sizeof(float));
    float* raw_ptr_2 = reinterpret_cast<float*>(raw_ptr + len);

    thrust::device_ptr<int> dev_ptr_int(raw_ptr);
    thrust::device_ptr<float> dev_ptr_float(raw_ptr_2);
    thrust::copy(lengths.begin(), lengths.end(), dev_ptr_int);
    thrust::copy(output.begin(), output.end(), dev_ptr_float);

    out_size = len;
    return raw_ptr;
}

} /* namespace ddj */
