/*
 * scale_encoding.cpp
 *
 *  Created on: 24-04-2015
 *      Author: Karol Dzitkowski
 */

#include "scale_encoding.h"
#include "scale.cuh"

namespace ddj
{

template<typename T>
SharedCudaPtr<char> ScaleEncoding::Encode(SharedCudaPtr<T> data, ScaleEncodingMetadata<T>& metadata)
{
	return scaleEncode(data, metadata.min);
}

template<typename T>
SharedCudaPtr<T> ScaleEncoding::Decode(SharedCudaPtr<char> data, ScaleEncodingMetadata<T> metadata)
{
	return scaleDecode(data, metadata.min);
}

template SharedCudaPtr<char> ScaleEncoding::Encode<float>(SharedCudaPtr<float> data, ScaleEncodingMetadata<float>& metadata);
template SharedCudaPtr<float> ScaleEncoding::Decode<float>(SharedCudaPtr<char> data, ScaleEncodingMetadata<float> metadata);


} /* namespace ddj */
