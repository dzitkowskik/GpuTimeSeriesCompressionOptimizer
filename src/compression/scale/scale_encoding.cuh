/*
 * scale_encoding.h
 *
 *  Created on: 24-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_SCALE_ENCODING_H_
#define DDJ_SCALE_ENCODING_H_

#include "core/cuda_ptr.h"

namespace ddj
{

class ScaleEncoding
{
public:
	template<typename T> SharedCudaPtr<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtr<char> data);
};

} /* namespace ddj */
#endif /* DDJ_SCALE_ENCODING_H_ */
