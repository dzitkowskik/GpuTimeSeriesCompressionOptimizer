/*
 *  cuda_array_transform.hpp
 *
 *  Created on: 09-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_UTIL_CUDA_ARRAY_TRANSFORM_HPP_
#define DDJ_UTIL_CUDA_ARRAY_TRANSFORM_HPP_

#include "core/cuda_ptr.hpp"
#include "core/execution_policy.hpp"

namespace ddj
{

class CudaArrayTransform
{
public:
    template<typename T, typename UnaryOperator>
    void TransformInPlace(SharedCudaPtr<T> data, UnaryOperator op);

    template<typename T, typename UnaryOperator>
    SharedCudaPtr<T> Transform(SharedCudaPtr<T> data, UnaryOperator op);

private:
    ExecutionPolicy _policy;
};

} /* namespace ddj */
#endif /* DDJ_UTIL_CUDA_ARRAY_TRANSFORM_HPP_ */
