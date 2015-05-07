/*
 * scale.cuh
 *
 *  Created on: 24-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef SCALE_CUH_
#define SCALE_CUH_

#include "core/cuda_ptr.h"

namespace ddj
{

template<typename T> SharedCudaPtr<char> scaleEncode(SharedCudaPtr<T> data, T& min);
template<typename T> SharedCudaPtr<T> scaleDecode(SharedCudaPtr<char> data, T& min);

} /* namespace ddj */
#endif /* SCALE_CUH_ */
