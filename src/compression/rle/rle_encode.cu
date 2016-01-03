/*
 *  thrust_rle.cu
 *
 *  Created on: 18-04-2015
 *      Author: Karol Dzitkowski
 */

#include "compression/rle/rle_encoding.hpp"
#include "helpers/helper_print.hpp"
#include "core/macros.h"

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
    auto resultLengths = CudaPtr<int>::make_shared(len);
    auto resultValues = CudaPtr<T>::make_shared(len);

    resultLengths->fillFromHost(lengths.data().get(), len);
    resultValues->fillFromHost(output.data().get(), len);

    return SharedCudaPtrVector<char> {
    	metadata,
    	CastSharedCudaPtr<int, char>(resultLengths),
    	CastSharedCudaPtr<T, char>(resultValues)
    };
}

size_t RleEncoding::GetCompressedSize(SharedCudaPtr<char> data, DataType type)
{
	if(data->size() <= 0) return 0;
	switch(type)
	{
		case DataType::d_int:
			return GetCompressedSize(CastSharedCudaPtr<char, int>(data));
		case DataType::d_float:
			return GetCompressedSize(CastSharedCudaPtr<char, float>(data));
		default:
			throw NotImplementedException("No DictEncoding::GetCompressedSize implementation for that type");
	}
}

template<typename T>
size_t RleEncoding::GetCompressedSize(SharedCudaPtr<T> data)
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
	return len * sizeof(int) + len * sizeof(T);
}

#define RLE_ENCODE_SPEC(X) \
    template SharedCudaPtrVector<char> RleEncoding::Encode<X>(SharedCudaPtr<X> data);
FOR_EACH(RLE_ENCODE_SPEC, char, short, double, float, int, long, long long, unsigned int)

} /* namespace ddj */
