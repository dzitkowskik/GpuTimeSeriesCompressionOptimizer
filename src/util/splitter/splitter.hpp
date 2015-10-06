/*
 *  splitter.hpp
 *
 *  Created on: 05-10-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_UTIL_SPLITTER_HPP_
#define DDJ_UTIL_SPLITTER_HPP_

#include "core/cuda_ptr.hpp"
#include "core/execution_policy.hpp"

namespace ddj
{

class Splitter
{
public:
	template<typename T> SharedCudaPtrTuple<T>
	Split(SharedCudaPtr<T> data, SharedCudaPtr<int> stencil);

	template<typename T> SharedCudaPtrTuple<T>
	Split2(SharedCudaPtr<T> data, SharedCudaPtr<int> stencil);

	template<typename T> SharedCudaPtr<T>
	CopyIf(SharedCudaPtr<T> data, SharedCudaPtr<int> stencil);

	template<typename T> SharedCudaPtr<T>
	CopyIfNot(SharedCudaPtr<T> data, SharedCudaPtr<int> stencil);

	template<typename T> SharedCudaPtr<T>
	Merge(SharedCudaPtrTuple<T> data, SharedCudaPtr<int> stencil);

private:
	SharedCudaPtr<int> NegateStencil(SharedCudaPtr<int> stencil);

private:
    ExecutionPolicy _policy;
};

} /* namespace ddj */
#endif /* DDJ_UTIL_SPLITTER_HPP_ */
