#include "stencil.hpp"
#include "helpers/helper_cuda.cuh"
#include "core/cuda_macros.cuh"
#include "core/macros.h"

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

}/* namespace ddj */
