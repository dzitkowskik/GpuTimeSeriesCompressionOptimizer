#include "stencil.hpp"
#include "helpers/helper_cuda.cuh"

namespace ddj {

__global__ void packKernel(int* data, int dataSize, char* output, int outputSize)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // char array index
    unsigned int input_idx_start = idx * 8;
	if(idx >= outputSize) return;
    char part = 0xFF, x = 0xFF;

    #pragma unroll
    for(int i = 0; i < 8; i++)
    {
        idx = input_idx_start + i;
        idx = idx < dataSize ? idx : dataSize - 1;
        x = (char)data[idx];
        part ^= (-x ^ part) & (1 << i);
    }

    output[idx] = part;
}

__global__ void unpackKernel(char* data, int dataSize, int* output, int outputSize)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // char array index
    unsigned int output_idx_start = idx * 8;
    if(idx >= dataSize) return;
    char number = data[idx];
    int bit = 0xFF;

    // from now idx is the output index
    #pragma unroll
    for(int i = 0; i < 8; i++)
    {
        bit = (number >> i) & 1;
        idx = output_idx_start + i;
        idx = idx < outputSize ? idx : outputSize - 1;
        output[idx] = bit;
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
    auto result = CudaPtr<int>::make_shared(numElements);

    this->_policy.setSize(data->size());
    cudaLaunch(this->_policy, unpackKernel,
        data->get(), data->size(), result->get(), result->size());

    return Stencil(result);
}

}/* namespace ddj */
