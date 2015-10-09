/*
 *  rle_decode.cu
 *
 *  Created on: 18-04-2015
 *      Author: Karol Dzitkowski
 */

#include "compression/rle/rle_encoding.hpp"
#include "helpers/helper_macros.h"

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>

namespace ddj {

template<typename T>
SharedCudaPtr<T> RleEncoding::Decode(SharedCudaPtrVector<char> input)
{
	// GET LENGTH FROM METADATA
	int length;
	auto metadata = input[0];
	CUDA_CALL( cudaMemcpy(&length, metadata->get(), sizeof(int), CPY_DTH) );

	// PREPARE INPUT DATA
	auto data = input[1];
    int* lengths = reinterpret_cast<int*>(data->get());
    T* values = reinterpret_cast<T*>(lengths + length);
    thrust::device_ptr<int> lengthsPtr(lengths);
    thrust::device_ptr<T> valuesPtr(values);
    thrust::device_vector<int> lengthsVector(lengthsPtr, lengthsPtr + length);
    thrust::device_vector<T> valuesVector(valuesPtr, valuesPtr + length);

    // SCAN LENGTHS
    thrust::inclusive_scan(lengthsVector.begin(), lengthsVector.end(), lengthsVector.begin());

    // output size is sum of the run lengths
    int n = lengthsVector.back();

    // compute input index for each output element
    thrust::device_vector<int> indices(n);
    thrust::lower_bound(
		lengthsVector.begin(),
		lengthsVector.end(),
        thrust::counting_iterator<int>(1),
        thrust::counting_iterator<int>(n + 1),
        indices.begin());

    // gather input elements
    auto result = CudaPtr<T>::make_shared(n);
    thrust::device_ptr<T> resultPtr(result->get());
    thrust::gather(indices.begin(), indices.end(), valuesVector.begin(), resultPtr);

    return result;
}

#define SCALE_DECODE_SPEC(X) \
    template SharedCudaPtr<X> RleEncoding::Decode<X>(SharedCudaPtrVector<char>);
FOR_EACH(SCALE_DECODE_SPEC, float, int, long long, unsigned int)

} /* namespace ddj */
