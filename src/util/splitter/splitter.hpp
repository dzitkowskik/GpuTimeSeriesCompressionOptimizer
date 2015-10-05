/*
 *  splitter.hpp
 *
 *  Created on: 05-10-2015
 *      Author: Karol Dzitkowski
 */

#ifndef HELPER_UTIL_SPLITTER_HPP_
#define HELPER_UTIL_SPLITTER_HPP_

#include "core/cuda_ptr.hpp"
#include "core/execution_policy.hpp"

namespace ddj
{

class Splitter
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

private:
	SharedCudaPtr<int> NegateStencil(SharedCudaPtr<int> stencil);

private:
    ExecutionPolicy _policy;
};

} /* namespace ddj */
#endif /* HELPER_UTIL_SPLITTER_HPP_ */
