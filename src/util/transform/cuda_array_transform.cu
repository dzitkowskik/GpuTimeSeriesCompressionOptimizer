/*
 *  cuda_array_transform.cu
 *
 *  Created on: 10-05-2015
 *      Author: Karol Dzitkowski
 */

#include "util/transform/cuda_array_transform.hpp"
#include "core/operators.cuh"
#include "helpers/helper_cuda.cuh"
#include "core/macros.h"

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

template<typename InputType, typename OutputType, typename UnaryOperator>
__global__ void transformKernel(const InputType* data, int size, UnaryOperator op, OutputType* output)
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

template<typename InputType, typename OutputType, typename UnaryOperator>
SharedCudaPtr<OutputType> CudaArrayTransform::Transform(SharedCudaPtr<InputType> data, UnaryOperator op)
{
	auto result = CudaPtr<OutputType>::make_shared(data->size());

    this->_policy.setSize(data->size());
    cudaLaunch(this->_policy, transformKernel<InputType, OutputType, UnaryOperator>,
		data->get(),
        data->size(),
        op,
        result->get()
    );

    cudaDeviceSynchronize();
	return result;
}

template<typename InputType, typename OutputType>
__global__ void _castKernel(InputType* input, size_t size, OutputType* output)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	output[idx] = (OutputType)input[idx];
}

template<typename InputType, typename OutputType>
SharedCudaPtr<OutputType> CudaArrayTransform::Cast(SharedCudaPtr<InputType> data)
{
	auto result = CudaPtr<OutputType>::make_shared(data->size());

	this->_policy.setSize(data->size());
	cudaLaunch(this->_policy, _castKernel<InputType, OutputType>,
			data->get(),
			data->size(),
			result->get());
	cudaDeviceSynchronize();
	return result;
}

#define TRANSFORM_IN_PLACE_SPEC(X) \
	template void CudaArrayTransform::TransformInPlace<X, OutsideOperator<X,X>>(SharedCudaPtr<X>, OutsideOperator<X,X>); \
    template void CudaArrayTransform::TransformInPlace<X, InsideOperator<X,X>>(SharedCudaPtr<X>, InsideOperator<X,X> op); \
    template void CudaArrayTransform::TransformInPlace<X, AdditionOperator<X,X>>(SharedCudaPtr<X>, AdditionOperator<X,X> op); \
    template void CudaArrayTransform::TransformInPlace<X, SubtractionOperator<X,X>>(SharedCudaPtr<X>, SubtractionOperator<X,X> op); \
    template void CudaArrayTransform::TransformInPlace<X, MultiplicationOperator<X,X>>(SharedCudaPtr<X>, MultiplicationOperator<X,X> op); \
    template void CudaArrayTransform::TransformInPlace<X, AbsoluteOperator<X,X>>(SharedCudaPtr<X>, AbsoluteOperator<X,X> op); \
    template void CudaArrayTransform::TransformInPlace<X, ZeroOperator<X,X>>(SharedCudaPtr<X>, ZeroOperator<X,X> op); \
    template void CudaArrayTransform::TransformInPlace<X, OneOperator<X,X>>(SharedCudaPtr<X>, OneOperator<X,X> op); \
    template void CudaArrayTransform::TransformInPlace<X, NegateOperator<X,X>>(SharedCudaPtr<X>, NegateOperator<X,X> op); \
    template void CudaArrayTransform::TransformInPlace<X, FillOperator<X,X>>(SharedCudaPtr<X>, FillOperator<X,X> op); \
    template void CudaArrayTransform::TransformInPlace<X, DivisionOperator<X,X>>(SharedCudaPtr<X>, DivisionOperator<X,X> op);
FOR_EACH(TRANSFORM_IN_PLACE_SPEC, float, int, long, long long, unsigned int)

#define TRANS_MODULUS_SPEC(X) \
    template SharedCudaPtr<int> CudaArrayTransform::Transform<X, int, ModulusOperator<X, int>>(SharedCudaPtr<X>, ModulusOperator<X,int> op); \
    template void CudaArrayTransform::TransformInPlace<X, ModulusOperator<X,X>>(SharedCudaPtr<X>, ModulusOperator<X,X > op);
FOR_EACH(TRANS_MODULUS_SPEC, int, long, long long, unsigned int)

#define TRANSFORM_SPEC(X) \
		template SharedCudaPtr<int> CudaArrayTransform::Transform<X, int, OutsideOperator		<X, int>>(SharedCudaPtr<X>, OutsideOperator			<X, int> op); \
	    template SharedCudaPtr<int> CudaArrayTransform::Transform<X, int, InsideOperator		<X, int>>(SharedCudaPtr<X>, InsideOperator			<X, int> op); \
	    template SharedCudaPtr<int> CudaArrayTransform::Transform<X, int, AdditionOperator		<X, int>>(SharedCudaPtr<X>, AdditionOperator		<X, int> op); \
	    template SharedCudaPtr<int> CudaArrayTransform::Transform<X, int, SubtractionOperator	<X, int>>(SharedCudaPtr<X>, SubtractionOperator		<X, int> op); \
	    template SharedCudaPtr<int> CudaArrayTransform::Transform<X, int, MultiplicationOperator<X, int>>(SharedCudaPtr<X>, MultiplicationOperator	<X, int> op); \
	    template SharedCudaPtr<int> CudaArrayTransform::Transform<X, int, AbsoluteOperator		<X, int>>(SharedCudaPtr<X>, AbsoluteOperator		<X, int> op); \
	    template SharedCudaPtr<int> CudaArrayTransform::Transform<X, int, ZeroOperator			<X, int>>(SharedCudaPtr<X>, ZeroOperator			<X, int> op); \
	    template SharedCudaPtr<int> CudaArrayTransform::Transform<X, int, OneOperator			<X, int>>(SharedCudaPtr<X>, OneOperator				<X, int> op); \
	    template SharedCudaPtr<int> CudaArrayTransform::Transform<X, int, NegateOperator		<X, int>>(SharedCudaPtr<X>, NegateOperator			<X, int> op); \
	    template SharedCudaPtr<int> CudaArrayTransform::Transform<X, int, FillOperator			<X, int>>(SharedCudaPtr<X>, FillOperator			<X, int> op); \
	    template SharedCudaPtr<int> CudaArrayTransform::Transform<X, int, DivisionOperator		<X, int>>(SharedCudaPtr<X>, DivisionOperator		<X, int> op); \
	    \
	    template SharedCudaPtr<float> CudaArrayTransform::Transform<X, float, OutsideOperator		<X, float>>(SharedCudaPtr<X>, OutsideOperator			<X, float> op); \
	    template SharedCudaPtr<float> CudaArrayTransform::Transform<X, float, InsideOperator		<X, float>>(SharedCudaPtr<X>, InsideOperator			<X, float> op); \
	    template SharedCudaPtr<float> CudaArrayTransform::Transform<X, float, AdditionOperator		<X, float>>(SharedCudaPtr<X>, AdditionOperator			<X, float> op); \
	    template SharedCudaPtr<float> CudaArrayTransform::Transform<X, float, SubtractionOperator	<X, float>>(SharedCudaPtr<X>, SubtractionOperator		<X, float> op); \
	    template SharedCudaPtr<float> CudaArrayTransform::Transform<X, float, MultiplicationOperator<X, float>>(SharedCudaPtr<X>, MultiplicationOperator	<X, float> op); \
	    template SharedCudaPtr<float> CudaArrayTransform::Transform<X, float, AbsoluteOperator		<X, float>>(SharedCudaPtr<X>, AbsoluteOperator			<X, float> op); \
	    template SharedCudaPtr<float> CudaArrayTransform::Transform<X, float, ZeroOperator			<X, float>>(SharedCudaPtr<X>, ZeroOperator				<X, float> op); \
	    template SharedCudaPtr<float> CudaArrayTransform::Transform<X, float, OneOperator			<X, float>>(SharedCudaPtr<X>, OneOperator				<X, float> op); \
	    template SharedCudaPtr<float> CudaArrayTransform::Transform<X, float, NegateOperator		<X, float>>(SharedCudaPtr<X>, NegateOperator			<X, float> op); \
	    template SharedCudaPtr<float> CudaArrayTransform::Transform<X, float, FillOperator			<X, float>>(SharedCudaPtr<X>, FillOperator				<X, float> op); \
	    template SharedCudaPtr<float> CudaArrayTransform::Transform<X, float, DivisionOperator		<X, float>>(SharedCudaPtr<X>, DivisionOperator			<X, float> op); \
	   	\
	   	template SharedCudaPtr<double> CudaArrayTransform::Transform<X, double, OutsideOperator			<X, double>>(SharedCudaPtr<X>, OutsideOperator			<X, double> op); \
	    template SharedCudaPtr<double> CudaArrayTransform::Transform<X, double, InsideOperator			<X, double>>(SharedCudaPtr<X>, InsideOperator			<X, double> op); \
	    template SharedCudaPtr<double> CudaArrayTransform::Transform<X, double, AdditionOperator		<X, double>>(SharedCudaPtr<X>, AdditionOperator			<X, double> op); \
	    template SharedCudaPtr<double> CudaArrayTransform::Transform<X, double, SubtractionOperator		<X, double>>(SharedCudaPtr<X>, SubtractionOperator		<X, double> op); \
	    template SharedCudaPtr<double> CudaArrayTransform::Transform<X, double, MultiplicationOperator	<X, double>>(SharedCudaPtr<X>, MultiplicationOperator	<X, double> op); \
	    template SharedCudaPtr<double> CudaArrayTransform::Transform<X, double, AbsoluteOperator		<X, double>>(SharedCudaPtr<X>, AbsoluteOperator			<X, double> op); \
	    template SharedCudaPtr<double> CudaArrayTransform::Transform<X, double, ZeroOperator			<X, double>>(SharedCudaPtr<X>, ZeroOperator				<X, double> op); \
	    template SharedCudaPtr<double> CudaArrayTransform::Transform<X, double, OneOperator				<X, double>>(SharedCudaPtr<X>, OneOperator				<X, double> op); \
	    template SharedCudaPtr<double> CudaArrayTransform::Transform<X, double, NegateOperator			<X, double>>(SharedCudaPtr<X>, NegateOperator			<X, double> op); \
	    template SharedCudaPtr<double> CudaArrayTransform::Transform<X, double, FillOperator			<X, double>>(SharedCudaPtr<X>, FillOperator				<X, double> op); \
	    template SharedCudaPtr<double> CudaArrayTransform::Transform<X, double, DivisionOperator		<X, double>>(SharedCudaPtr<X>, DivisionOperator			<X, double> op); \
	    \
	    template SharedCudaPtr<long int> CudaArrayTransform::Transform<X, long int, OutsideOperator			<X, long int>>(SharedCudaPtr<X>, OutsideOperator		<X, long int> op); \
	    template SharedCudaPtr<long int> CudaArrayTransform::Transform<X, long int, InsideOperator			<X, long int>>(SharedCudaPtr<X>, InsideOperator			<X, long int> op); \
	    template SharedCudaPtr<long int> CudaArrayTransform::Transform<X, long int, AdditionOperator		<X, long int>>(SharedCudaPtr<X>, AdditionOperator		<X, long int> op); \
	    template SharedCudaPtr<long int> CudaArrayTransform::Transform<X, long int, SubtractionOperator		<X, long int>>(SharedCudaPtr<X>, SubtractionOperator	<X, long int> op); \
	    template SharedCudaPtr<long int> CudaArrayTransform::Transform<X, long int, MultiplicationOperator	<X, long int>>(SharedCudaPtr<X>, MultiplicationOperator	<X, long int> op); \
	    template SharedCudaPtr<long int> CudaArrayTransform::Transform<X, long int, AbsoluteOperator		<X, long int>>(SharedCudaPtr<X>, AbsoluteOperator		<X, long int> op); \
	    template SharedCudaPtr<long int> CudaArrayTransform::Transform<X, long int, ZeroOperator			<X, long int>>(SharedCudaPtr<X>, ZeroOperator			<X, long int> op); \
	    template SharedCudaPtr<long int> CudaArrayTransform::Transform<X, long int, OneOperator				<X, long int>>(SharedCudaPtr<X>, OneOperator			<X, long int> op); \
	    template SharedCudaPtr<long int> CudaArrayTransform::Transform<X, long int, NegateOperator			<X, long int>>(SharedCudaPtr<X>, NegateOperator			<X, long int> op); \
	    template SharedCudaPtr<long int> CudaArrayTransform::Transform<X, long int, FillOperator			<X, long int>>(SharedCudaPtr<X>, FillOperator			<X, long int> op); \
	    template SharedCudaPtr<long int> CudaArrayTransform::Transform<X, long int, DivisionOperator		<X, long int>>(SharedCudaPtr<X>, DivisionOperator		<X, long int> op); \
	    \
	    template SharedCudaPtr<long long int> CudaArrayTransform::Transform<X, long long int, OutsideOperator		<X, long long int>>(SharedCudaPtr<X>, OutsideOperator		<X, long long int> op); \
	    template SharedCudaPtr<long long int> CudaArrayTransform::Transform<X, long long int, InsideOperator		<X, long long int>>(SharedCudaPtr<X>, InsideOperator		<X, long long int> op); \
	    template SharedCudaPtr<long long int> CudaArrayTransform::Transform<X, long long int, AdditionOperator		<X, long long int>>(SharedCudaPtr<X>, AdditionOperator		<X, long long int> op); \
	    template SharedCudaPtr<long long int> CudaArrayTransform::Transform<X, long long int, SubtractionOperator	<X, long long int>>(SharedCudaPtr<X>, SubtractionOperator	<X, long long int> op); \
	    template SharedCudaPtr<long long int> CudaArrayTransform::Transform<X, long long int, MultiplicationOperator<X, long long int>>(SharedCudaPtr<X>, MultiplicationOperator<X, long long int> op); \
	    template SharedCudaPtr<long long int> CudaArrayTransform::Transform<X, long long int, AbsoluteOperator		<X, long long int>>(SharedCudaPtr<X>, AbsoluteOperator		<X, long long int> op); \
	    template SharedCudaPtr<long long int> CudaArrayTransform::Transform<X, long long int, ZeroOperator			<X, long long int>>(SharedCudaPtr<X>, ZeroOperator			<X, long long int> op); \
	    template SharedCudaPtr<long long int> CudaArrayTransform::Transform<X, long long int, OneOperator			<X, long long int>>(SharedCudaPtr<X>, OneOperator			<X, long long int> op); \
	    template SharedCudaPtr<long long int> CudaArrayTransform::Transform<X, long long int, NegateOperator		<X, long long int>>(SharedCudaPtr<X>, NegateOperator		<X, long long int> op); \
	    template SharedCudaPtr<long long int> CudaArrayTransform::Transform<X, long long int, FillOperator			<X, long long int>>(SharedCudaPtr<X>, FillOperator			<X, long long int> op); \
	    template SharedCudaPtr<long long int> CudaArrayTransform::Transform<X, long long int, DivisionOperator		<X, long long int>>(SharedCudaPtr<X>, DivisionOperator		<X, long long int> op);
FOR_EACH(TRANSFORM_SPEC, float, int, long, long long, unsigned int)



} /* namespace ddj */
