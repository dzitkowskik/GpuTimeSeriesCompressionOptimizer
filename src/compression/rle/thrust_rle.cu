/*
 *  thrust_rle.cu
 *
 *  Created on: 18-04-2015
 *      Author: Karol Dzitkowski
 */

#include "thrust_rle.cuh"
#include "helpers/helper_print.hpp"
#include "helpers/helper_macros.h"

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <iostream>
#include <iterator>
#include <thrust/reduce.h>
#include <thrust/system_error.h>

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

template<typename T>
void* ThrustRleCompression::Encode(T* data, const int in_size, int& out_size)
{
    thrust::device_ptr<T> d_ptr(data);
    thrust::device_vector<T> input(d_ptr, d_ptr + in_size);
    thrust::device_vector<T> output(in_size);
    thrust::device_vector<int>  lengths(in_size);

    // compute run lengths
    auto result = thrust::reduce_by_key(
        input.begin(),
        input.end(),
        thrust::constant_iterator<int>(1),
        output.begin(),
        lengths.begin());

    int len = result.first - output.begin();

    #if DDJ_THRUST_RLE_DEBUG
        printInputData(input);
        printOutputData(output, lengths);
    #endif

    // prepare data
    int* raw_ptr;
    int compressed_size = len * sizeof(int) + len * sizeof(T);
    CUDA_CALL( cudaMalloc((void **) &raw_ptr, compressed_size) );
    T* raw_ptr_2 = reinterpret_cast<T*>(raw_ptr + len);
    thrust::device_ptr<int> dev_ptr_int(raw_ptr);
    thrust::device_ptr<T> dev_ptr_float(raw_ptr_2);
    thrust::copy(lengths.begin(), lengths.begin()+len, dev_ptr_int);
    thrust::copy(output.begin(), output.begin()+len, dev_ptr_float);

    out_size = len;
    return raw_ptr;
}


#define SCALE_SPEC(X) \
    template void* ThrustRleCompression::Encode<X>(X* data, const int in_size, int& out_size); \
    template X* ThrustRleCompression::Decode<X>(void* data, int in_size, int& out_size);
FOR_EACH(SCALE_SPEC, double, float, int, long, long long, unsigned int, unsigned long, unsigned long long)

} /* namespace ddj */
