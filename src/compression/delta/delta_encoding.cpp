/*
 * delta_encoding.cpp
 *
 *  Created on: 18-04-2015
 *      Author: Karol Dzitkowski
 */

#include "delta_encoding.h"
#include "delta.cuh"

namespace ddj
{

template<typename T>
void* Encode(T* data, int in_size, int& out_size, DeltaEncodingMetadata<T>& metadata)
{
	out_size = in_size - 1;
	return deltaEncode(data, in_size, metadata.first);
}

template<typename T>
T* Decode(void* data, int in_size, int& out_size, DeltaEncodingMetadata<T> metadata)
{
	out_size = in_size + 1;
	return deltaDecode(data, out_size, metadata.first);
}

} /* namespace ddj */
