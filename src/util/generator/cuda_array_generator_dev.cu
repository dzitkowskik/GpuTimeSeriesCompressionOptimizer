#include "cuda_array_generator.hpp"
#include "helpers/helper_cuda.cuh"
#include "core/macros.h"

namespace ddj {

template<typename T>
__global__ void _createConsecutiveNumbersArrayKernel(T* data, int size, T start)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	data[idx] = start + idx;
}

template<typename T>
__global__ void _createConsecutiveNumbersArrayWithStepKernel(T* data, int size, T start, T step)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	data[idx] = start + (idx * step);
}

template<typename T> SharedCudaPtr<T>
CudaArrayGenerator::CreateConsecutiveNumbersArray(int size, T start)
{
    auto result = CudaPtr<T>::make_shared(size);

    this->_policy.setSize(size);
    cudaLaunch(this->_policy, _createConsecutiveNumbersArrayKernel<T>,
        result->get(),
        size,
        start
    );

    cudaDeviceSynchronize();
    return result;
}

template<typename T> SharedCudaPtr<T>
CudaArrayGenerator::CreateConsecutiveNumbersArray(int size, T start, T step)
{
    auto result = CudaPtr<T>::make_shared(size);

    this->_policy.setSize(size);
    cudaLaunch(this->_policy, _createConsecutiveNumbersArrayWithStepKernel<T>,
        result->get(),
        size,
        start,
        step
    );

    cudaDeviceSynchronize();
    return result;
}

__global__ void _setPrecisionKernel(float* data, size_t size, int* precision)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	int prec = precision[idx];
	int mul = 1;
	while(prec--) mul *= 10;
	data[idx] = (float)(int)(data[idx]*mul);
	data[idx] /= mul;
}

SharedCudaPtr<float> CudaArrayGenerator::CreateRandomFloatsWithMaxPrecision(int size, int maxPrecision)
{
	auto randomFloats = this->GenerateRandomFloatDeviceArray(size);
	auto randomPrecison = this->GenerateRandomIntDeviceArray(size, 0, maxPrecision+1);

	this->_policy.setSize(size);
	cudaLaunch(this->_policy, _setPrecisionKernel,
		randomFloats->get(),
		size,
		randomPrecison->get()
	);

	cudaDeviceSynchronize();
	return randomFloats;
}

#define CUDA_ARRAY_GENERATOR_SPEC(X) \
	template SharedCudaPtr<X> CudaArrayGenerator::CreateConsecutiveNumbersArray<X>(int, X); \
	template SharedCudaPtr<X> CudaArrayGenerator::CreateConsecutiveNumbersArray<X>(int, X, X);
FOR_EACH(CUDA_ARRAY_GENERATOR_SPEC, float, int, long, long long, unsigned int)

} /* namespace ddj */
