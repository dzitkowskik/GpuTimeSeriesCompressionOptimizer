/*
 * cuda_array_generator.hpp 26-03-2015 Karol Dzitkowski
 */

#ifndef DDJ_UTIL_CUDA_ARRAY_GENERATOR_HPP_
#define DDJ_UTIL_CUDA_ARRAY_GENERATOR_HPP_

#include "core/cuda_ptr.hpp"
#include "util/transform/cuda_array_transform.hpp"
#include "core/execution_policy.hpp"

#include <boost/noncopyable.hpp>
#include <curand.h>

namespace ddj
{

class CudaArrayGenerator : private boost::noncopyable
{
public:
    CudaArrayGenerator();
    ~CudaArrayGenerator();

public:
    // TODO: translate to templates
    SharedCudaPtr<float> GenerateRandomFloatDeviceArray(int size);
    SharedCudaPtr<int> GenerateRandomIntDeviceArray(int size);
    SharedCudaPtr<int> GenerateRandomIntDeviceArray(int size, int from, int to);
    SharedCudaPtr<int> GenerateConsecutiveIntDeviceArray(int size);
    SharedCudaPtr<int> GenerateRandomStencil(int size);

    template<typename T> SharedCudaPtr<T> CreateConsecutiveNumbersArray(int size, T start);
    template<typename T> SharedCudaPtr<T> CreateConsecutiveNumbersArray(int size, T start, T step);

    SharedCudaPtr<float> CreateRandomFloatsWithMaxPrecision(int size, int maxPrecision);

private:
    curandGenerator_t _gen;
    CudaArrayTransform _transform;
    ExecutionPolicy _policy;
};

} /* namespace ddj */
#endif /* DDJ_UTIL_CUDA_ARRAY_GENERATOR_HPP_ */
