/*
 *  thrust_rle.cu
 *
 *  Created on: 18-04-2015
 *      Author: Karol Dzitkowski
 */

#include "compression/rle/rle_encoding.hpp"
#include "helpers/helper_print.hpp"
#include "helpers/helper_macros.h"

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>

namespace ddj {

template<typename T>
SharedCudaPtrVector<char> RleEncoding::Encode(SharedCudaPtr<T> data)
{
    thrust::device_ptr<T> d_ptr(data->get());
    thrust::device_vector<T> input(d_ptr, d_ptr + data->size());
    thrust::device_vector<T> output(data->size());
    thrust::device_vector<int>  lengths(data->size());

    // compute run lengths
    auto reduceResult = thrust::reduce_by_key(
        input.begin(),
        input.end(),
        thrust::constant_iterator<int>(1),
        output.begin(),
        lengths.begin());
    // get true output length
    int len = reduceResult.first - output.begin();

    // prepare metadata result
    auto metadata = CudaPtr<char>::make_shared(sizeof(int));
    metadata->fillFromHost((char*)&len, sizeof(int));

    // prepare data result
    int outputSize = len * sizeof(int) + len * sizeof(T);
    auto result = CudaPtr<char>::make_shared(outputSize);

    CUDA_CALL( cudaMemcpy(result->get(), lengths.data().get(), len*sizeof(int), CPY_DTD) );
    auto resultDataPtr = result->get()+(len*sizeof(int));
    CUDA_CALL( cudaMemcpy(resultDataPtr, output.data().get(), len*sizeof(T), CPY_DTD) );

    return SharedCudaPtrVector<char> {metadata, result};
}

#define RLE_ENCODE_SPEC(X) \
    template SharedCudaPtrVector<char> RleEncoding::Encode<X>(SharedCudaPtr<X> data);
FOR_EACH(RLE_ENCODE_SPEC, float, int, long long, unsigned int)

} /* namespace ddj */
