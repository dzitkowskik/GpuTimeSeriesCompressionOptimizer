/*
 *  cuda_array_transform.cu
 *
 *  Created on: 10-05-2015
 *      Author: Karol Dzitkowski
 */

#include "util/transform/cuda_array_transform.hpp"
#include "core/operators.cuh"
#include "helpers/helper_cuda.cuh"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>

namespace ddj
{

template<typename T, typename UnaryOperator>
__global__ void transformInPlaceKernel(T* data, int size, UnaryOperator op)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	data[idx] = op(data[idx]);
}

template<typename T, typename UnaryOperator>
__global__ void transformKernel(const T* data, int size, UnaryOperator op, T* output)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	output[idx] = op(data[idx]);
}

template<typename T, typename UnaryOperator>
void CudaArrayTransform::TransformInPlace(SharedCudaPtr<T> data, UnaryOperator op)
{
	this->_policy.setSize(data->size());
	cudaLaunch(this->_policy, transformInPlaceKernel<T, UnaryOperator>,
		data->get(),
        data->size(),
        op
    );

	cudaDeviceSynchronize();
}

template<typename T, typename UnaryOperator>
SharedCudaPtr<T> CudaArrayTransform::Transform(SharedCudaPtr<T> data, UnaryOperator op)
{
	auto result = CudaPtr<T>::make_shared(data->size());

    this->_policy.setSize(data->size());
    cudaLaunch(this->_policy, transformKernel<T, UnaryOperator>,
		data->get(),
        data->size(),
        op,
        result->get()
    );

    cudaDeviceSynchronize();
	return result;
}

#define TRANSFORM_SPEC(X) \
	template SharedCudaPtr<X> CudaArrayTransform::Transform<X, OutsideOperator<X>>(SharedCudaPtr<X>, OutsideOperator<X> op); \
    template SharedCudaPtr<X> CudaArrayTransform::Transform<X, InsideOperator<X>>(SharedCudaPtr<X>, InsideOperator<X> op); \
    template SharedCudaPtr<X> CudaArrayTransform::Transform<X, AdditionOperator<X>>(SharedCudaPtr<X>, AdditionOperator<X> op); \
    template SharedCudaPtr<X> CudaArrayTransform::Transform<X, SubtractionOperator<X>>(SharedCudaPtr<X>, SubtractionOperator<X> op); \
    template SharedCudaPtr<X> CudaArrayTransform::Transform<X, MultiplicationOperator<X>>(SharedCudaPtr<X>, MultiplicationOperator<X> op); \
    template SharedCudaPtr<X> CudaArrayTransform::Transform<X, AbsoluteOperator<X>>(SharedCudaPtr<X>, AbsoluteOperator<X> op); \
    template SharedCudaPtr<X> CudaArrayTransform::Transform<X, ZeroOperator<X>>(SharedCudaPtr<X>, ZeroOperator<X> op); \
    template SharedCudaPtr<X> CudaArrayTransform::Transform<X, OneOperator<X>>(SharedCudaPtr<X>, OneOperator<X> op); \
    template SharedCudaPtr<X> CudaArrayTransform::Transform<X, NegateOperator<X>>(SharedCudaPtr<X>, NegateOperator<X> op); \
    template SharedCudaPtr<X> CudaArrayTransform::Transform<X, FillOperator<X>>(SharedCudaPtr<X>, FillOperator<X> op);
FOR_EACH(TRANSFORM_SPEC, double, float, int, unsigned int, long, unsigned long, long long int, unsigned long long int)

#define TRANSFORM_IN_PLACE_SPEC(X) \
	template void CudaArrayTransform::TransformInPlace<X, OutsideOperator<X>>(SharedCudaPtr<X>, OutsideOperator<X> op); \
    template void CudaArrayTransform::TransformInPlace<X, InsideOperator<X>>(SharedCudaPtr<X>, InsideOperator<X> op); \
    template void CudaArrayTransform::TransformInPlace<X, AdditionOperator<X>>(SharedCudaPtr<X>, AdditionOperator<X> op); \
    template void CudaArrayTransform::TransformInPlace<X, SubtractionOperator<X>>(SharedCudaPtr<X>, SubtractionOperator<X> op); \
    template void CudaArrayTransform::TransformInPlace<X, MultiplicationOperator<X>>(SharedCudaPtr<X>, MultiplicationOperator<X> op); \
    template void CudaArrayTransform::TransformInPlace<X, AbsoluteOperator<X>>(SharedCudaPtr<X>, AbsoluteOperator<X> op); \
    template void CudaArrayTransform::TransformInPlace<X, ZeroOperator<X>>(SharedCudaPtr<X>, ZeroOperator<X> op); \
    template void CudaArrayTransform::TransformInPlace<X, OneOperator<X>>(SharedCudaPtr<X>, OneOperator<X> op); \
    template void CudaArrayTransform::TransformInPlace<X, NegateOperator<X>>(SharedCudaPtr<X>, NegateOperator<X> op); \
    template void CudaArrayTransform::TransformInPlace<X, FillOperator<X>>(SharedCudaPtr<X>, FillOperator<X> op);
FOR_EACH(TRANSFORM_IN_PLACE_SPEC, double, float, int, unsigned int, long, unsigned long, long long int, unsigned long long int)

#define TRANS_MODULUS_SPEC(X) \
    template SharedCudaPtr<X> CudaArrayTransform::Transform<X, ModulusOperator<X>>(SharedCudaPtr<X>, ModulusOperator<X> op); \
    template void CudaArrayTransform::TransformInPlace<X, ModulusOperator<X>>(SharedCudaPtr<X>, ModulusOperator<X> op);
FOR_EACH(TRANS_MODULUS_SPEC, int, unsigned int, long, unsigned long, long long int, unsigned long long int)


} /* namespace ddj */
