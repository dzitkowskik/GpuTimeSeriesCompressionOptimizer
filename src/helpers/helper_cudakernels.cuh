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

template<class T>
using SharedCudaPtrTuple = std::tuple<SharedCudaPtr<T>, SharedCudaPtr<T>>;

class HelperCudaKernels
{
public:
	template<typename T> SharedCudaPtrTuple<T>
	SplitKernel(SharedCudaPtr<T> data, SharedCudaPtr<int> stencil);

	template<typename T> SharedCudaPtrTuple<T>
	SplitKernel2(SharedCudaPtr<T> data, SharedCudaPtr<int> stencil);

	template<typename T> SharedCudaPtr<T>
	CopyIfKernel(SharedCudaPtr<T> data, SharedCudaPtr<int> stencil);

	template<typename T> SharedCudaPtr<T>
	CopyIfNotKernel(SharedCudaPtr<T> data, SharedCudaPtr<int> stencil);
};

} /* namespace ddj */
#endif /* HELPER_CUDAKERNELS_CUH_ */
