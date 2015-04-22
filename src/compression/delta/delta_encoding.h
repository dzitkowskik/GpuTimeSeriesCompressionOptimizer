/*
 * delta_encoding.h
 *
 *  Created on: 18-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DELTA_ENCODING_H_
#define DELTA_ENCODING_H_

namespace ddj
{

template<typename T>
class DeltaEncodingMetadata
{
public:
	T first;
};

class DeltaEncoding
{
public:
	template<typename T>
    void* Encode(T* data, int in_size, int& out_size, DeltaEncodingMetadata<T>& metadata);
	template<typename T>
    T* Decode(void* data, int in_size, int& out_size, DeltaEncodingMetadata<T> metadata);
};

} /* namespace ddj */
#endif /* DELTA_ENCODING_H_ */
