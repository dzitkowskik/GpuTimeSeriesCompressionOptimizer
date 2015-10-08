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
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>

namespace ddj {

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

#define RLE_ENCODE_SPEC(X) \
    template void* ThrustRleCompression::Encode<X>(X* data, const int in_size, int& out_size);
FOR_EACH(RLE_ENCODE_SPEC, float, int, long long, unsigned int)

} /* namespace ddj */
