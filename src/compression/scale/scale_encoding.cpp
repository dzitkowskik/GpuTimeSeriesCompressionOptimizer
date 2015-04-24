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
void* ScaleEncoding::Encode(T* data, int in_size, int& out_size, ScaleEncodingMetadata<T>& metadata)
{
	out_size = in_size;
	return scaleEncode(data, in_size, metadata.min);
}

template<typename T>
T* ScaleEncoding::Decode(void* data, int in_size, int& out_size, ScaleEncodingMetadata<T> metadata)
{
	out_size = in_size;
	return scaleDecode((float*)data, in_size, metadata.min);
}

template void* ScaleEncoding::Encode<float>(float* data, int in_size, int& out_size, ScaleEncodingMetadata<float>& metadata);
template float* ScaleEncoding::Decode<float>(void* data, int in_size, int& out_size, ScaleEncodingMetadata<float> metadata);


} /* namespace ddj */
