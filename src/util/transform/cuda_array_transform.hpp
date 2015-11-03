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
#include "transform_operators.hpp"

namespace ddj
{

class CudaArrayTransform
{
public:
    template<typename T, typename UnaryOperator>
    void TransformInPlace(SharedCudaPtr<T> data, UnaryOperator op);

    template<typename InputType, typename OutputType, typename UnaryOperator>
    SharedCudaPtr<OutputType> Transform(SharedCudaPtr<InputType> data, UnaryOperator op);

    template<typename InputType, typename OutputType>
    SharedCudaPtr<OutputType> Cast(SharedCudaPtr<InputType> data);

private:
    ExecutionPolicy _policy;
};

} /* namespace ddj */
#endif /* DDJ_UTIL_CUDA_ARRAY_TRANSFORM_HPP_ */
