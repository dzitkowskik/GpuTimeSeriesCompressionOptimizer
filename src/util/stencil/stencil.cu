#include "stencil.hpp"
#include "helpers/helper_cuda.cuh"
#include "core/cuda_macros.cuh"

namespace ddj {

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
    auto result = CudaPtr<char>::make_shared(charsNeeded);

    this->_policy.setSize(charsNeeded);
    cudaLaunch(this->_policy, packKernel,
        this->_data->get(), dataSize, result->get(), result->size());

    return result;
}

Stencil Stencil::unpack(SharedCudaPtr<char> data, int numElements)
{
    ExecutionPolicy policy;
    auto result = CudaPtr<int>::make_shared(numElements);
    
    policy.setSize(data->size());
    cudaLaunch(policy, unpackKernel,
        data->get(), data->size(), result->get(), result->size());

    return Stencil(result);
}

}/* namespace ddj */
