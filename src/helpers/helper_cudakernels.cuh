/*
 * helper_cudakernels.cuh
 *
 *  Created on: 09-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef HELPER_CUDAKERNELS_CUH_
#define HELPER_CUDAKERNELS_CUH_

#include "core/cuda_ptr.hpp"

namespace ddj
{

class HelperCudaKernels
{
public:

	// Result 0 - Minimum, Result 1 - Maximum
	template<typename T> std::tuple<T,T>
	MinMax(SharedCudaPtr<T> data);
};

} /* namespace ddj */
#endif /* HELPER_CUDAKERNELS_CUH_ */
