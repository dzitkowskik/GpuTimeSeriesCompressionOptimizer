/*
 * scale_encoding.h
 *
 *  Created on: 24-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_SCALE_ENCODING_H_
#define DDJ_SCALE_ENCODING_H_

namespace ddj
{

template<typename T>
class ScaleEncodingMetadata
{
public:
	T min;
};

class ScaleEncoding
{
public:
	template<typename T>
    void* Encode(T* data, int in_size, int& out_size, ScaleEncodingMetadata<T>& metadata);
	template<typename T>
    T* Decode(void* data, int in_size, int& out_size, ScaleEncodingMetadata<T> metadata);
};

} /* namespace ddj */
#endif /* DDJ_SCALE_ENCODING_H_ */
