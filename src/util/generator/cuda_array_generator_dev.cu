#include "cuda_array_generator.hpp"
#include "helpers/helper_cuda.cuh"

namespace ddj {

template<typename T>
__global__ void createConsecutiveNumbersArrayKernel(T* data, int size, T start)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	data[idx] = start + idx;
}

template<typename T> SharedCudaPtr<T>
CudaArrayGenerator::CreateConsecutiveNumbersArray(int size, T start)
{
    auto result = CudaPtr<T>::make_shared(size);

    this->_policy.setSize(size);
    cudaLaunch(this->_policy, createConsecutiveNumbersArrayKernel<T>,
        result->get(),
        size,
        start
    );

    cudaDeviceSynchronize();
    return result;
}

#define CUDA_ARRAY_GENERATOR_SPEC(X) \
	template SharedCudaPtr<X> CudaArrayGenerator::CreateConsecutiveNumbersArray<X>(int, X);
FOR_EACH(CUDA_ARRAY_GENERATOR_SPEC, double, float, int, long int, long long int)

} /* namespace ddj */
