/*
 * delta_encoding.h
 *
 *  Created on: 18-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DELTA_ENCODING_H_
#define DELTA_ENCODING_H_

#include "core/cuda_ptr.h"

namespace ddj
{

class DeltaEncoding
{
public:
	template<typename T> SharedCudaPtr<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtr<char> data);
};

} /* namespace ddj */
#endif /* DELTA_ENCODING_H_ */
