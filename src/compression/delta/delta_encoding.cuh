/*
 * delta_encoding.cuh
 *
 *  Created on: 18-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DELTA_ENCODING_H_
#define DELTA_ENCODING_H_

#include "core/cuda_ptr.hpp"
#include "core/execution_policy.hpp"

namespace ddj {

class DeltaEncoding
{
	ExecutionPolicy policy;

public:
	template<typename T> SharedCudaPtr<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtr<char> data);
};

} /* namespace ddj */
#endif /* DELTA_ENCODING_H_ */
