#include "stencil.hpp"
#include "helpers/helper_cuda.cuh"
#include "core/cuda_macros.cuh"
#include "core/macros.h"
#include "stencil_operators.hpp"

namespace ddj {

Stencil::Stencil(SharedCudaPtr<char> data, int shift)
{
	this->_data = unpack(data, shift);
}

__global__ void packKernel(int* data, int dataSize, char* output, int outputSize)
{
    unsigned int output_idx = threadIdx.x + blockIdx.x * blockDim.x; // char array index
    unsigned int input_idx_start = output_idx * 8;
    unsigned int input_idx;
	if(output_idx >= outputSize) return;
    char part = 0;
    int number = 0;

    #pragma unroll
    for(int i = 0; i < 8; i++)
    {
    	input_idx = input_idx_start + i;
    	input_idx = input_idx < dataSize ? input_idx : dataSize - 1;
        number = data[input_idx];
        part = SetNthBit(i, number, part);
    }

    output[output_idx] = part;
}

__global__ void unpackKernel(char* data, int dataSize, int* output, int outputSize)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // char array index
    unsigned int output_idx_start = idx * 8;
    if(idx >= dataSize) return;
    char number = data[idx];

    // from now idx is the output index
    #pragma unroll
    for(int i = 0; i < 8; i++)
    {
        idx = output_idx_start + i;
        idx = idx < outputSize ? idx : outputSize - 1;
        output[idx] = GetNthBit(i, number) ? 1 : 0;
    }
}

SharedCudaPtr<char> Stencil::pack()
{
    int dataSize = this->_data->size();
    int charsNeeded = (dataSize + 7) / 8;
    auto result = CudaPtr<char>::make_shared(charsNeeded + 1);

    this->_policy.setSize(charsNeeded);
    cudaLaunch(this->_policy, packKernel,
        this->_data->get(), dataSize, result->get()+1, result->size()-1);

    char rest = dataSize % 8;
    CUDA_CALL( cudaMemcpy(result->get(), &rest, 1, CPY_HTD) );

    cudaDeviceSynchronize();

    return result;
}

SharedCudaPtr<int> Stencil::unpack(SharedCudaPtr<char> data, int shift)
{
	char* dataPtr = data->get() + shift;
	size_t size = data->size() - shift;

	// GET NUMBER OF ELEMENTS
	char rest;
	CUDA_CALL( cudaMemcpy(&rest, dataPtr, 1, CPY_DTH) );
	int numElements = (size-1)*8;
	if(rest) numElements -= 8 - rest;

	// PREAPARE MEMORY FOR RESULT
    auto result = CudaPtr<int>::make_shared(numElements);
    
	// UNPACK STENCIL
    this->_policy.setSize(size);
    cudaLaunch(this->_policy, unpackKernel,
        dataPtr+1,
        size-1,
        result->get(),
        result->size()
	);

    cudaDeviceSynchronize();

    return result;
}

template<typename T, typename Predicate>
__global__ void _createStencilKernel(T* data, size_t size, int* output, Predicate pred)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= size) return;
	output[idx] = pred(data[idx]);
}

template<typename T, typename Predicate>
Stencil Stencil::Create(SharedCudaPtr<T> data, Predicate pred)
{
	Stencil stencil(CudaPtr<int>::make_shared(data->size()));
	ExecutionPolicy policy;
	policy.setSize(data->size());
	cudaLaunch(policy, _createStencilKernel<T, Predicate>,
			data->get(),
			data->size(),
			stencil->get(),
			pred);
	cudaDeviceSynchronize();

	return stencil;
}

#define STENCIL_SPEC(X) \
	template Stencil Stencil::Create<X, EqualOperator<X>>(SharedCudaPtr<X> data, EqualOperator<X> pred); \
	template Stencil Stencil::Create<X, NotEqualOperator<X>>(SharedCudaPtr<X> data, NotEqualOperator<X> pred); \
	template Stencil Stencil::Create<X, InsideOperator<X>>(SharedCudaPtr<X> data, InsideOperator<X> pred); \
	\
	template Stencil Stencil::Create<X, OutsideOperator<float>>(SharedCudaPtr<X> data, OutsideOperator<float> pred); \
	template Stencil Stencil::Create<X, OutsideOperator<int>>(SharedCudaPtr<X> data, OutsideOperator<int> pred); \
	template Stencil Stencil::Create<X, OutsideOperator<long>>(SharedCudaPtr<X> data, OutsideOperator<long> pred); \
	template Stencil Stencil::Create<X, OutsideOperator<long long>>(SharedCudaPtr<X> data, OutsideOperator<long long> pred); \
	template Stencil Stencil::Create<X, OutsideOperator<unsigned int>>(SharedCudaPtr<X> data, OutsideOperator<unsigned int> pred); \
	\
	template Stencil Stencil::Create<X, LowerOperator<float>>(SharedCudaPtr<X> data, LowerOperator<float> pred); \
	template Stencil Stencil::Create<X, LowerOperator<int>>(SharedCudaPtr<X> data, LowerOperator<int> pred); \
	template Stencil Stencil::Create<X, LowerOperator<long>>(SharedCudaPtr<X> data, LowerOperator<long> pred); \
	template Stencil Stencil::Create<X, LowerOperator<long long>>(SharedCudaPtr<X> data, LowerOperator<long long> pred); \
	template Stencil Stencil::Create<X, LowerOperator<unsigned int>>(SharedCudaPtr<X> data, LowerOperator<unsigned int> pred);
FOR_EACH(STENCIL_SPEC, float, int, long, long long, unsigned int)

}/* namespace ddj */
