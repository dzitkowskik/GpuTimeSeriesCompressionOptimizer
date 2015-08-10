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

	template<typename T> SharedCudaPtr<T>
	ModuloKernel(SharedCudaPtr<T> data, T mod);

	template<typename T> void
	ModuloInPlaceKernel(SharedCudaPtr<T> data, T mod);

	template<typename T> SharedCudaPtr<T>
	AdditionKernel(SharedCudaPtr<T> data, T mod);

	template<typename T> void
	AdditionInPlaceKernel(SharedCudaPtr<T> data, T mod);

	template<typename T> SharedCudaPtr<T>
	CreateConsecutiveNumbersArray(int size, T start);

	// Result 0 - Minimum, Result 1 - Maximum
	template<typename T> std::tuple<T,T> MinMax(SharedCudaPtr<T> data);
};

} /* namespace ddj */
#endif /* HELPER_CUDAKERNELS_CUH_ */
