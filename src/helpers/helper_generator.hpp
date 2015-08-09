/*
 * helper_generator.hpp 26-03-2015 Karol Dzitkowski
 */

#ifndef DDJ_HELPER_GENERATOR_H_
#define DDJ_HELPER_GENERATOR_H_

#include "core/cuda_ptr.hpp"
#include <boost/noncopyable.hpp>
#include <curand.h>

namespace ddj
{

class HelperGenerator : private boost::noncopyable
{
public:
    HelperGenerator();
    ~HelperGenerator();

public:
    SharedCudaPtr<float> GenerateRandomFloatDeviceArray(int size);
    SharedCudaPtr<int> GenerateRandomIntDeviceArray(int size);
    SharedCudaPtr<int> GenerateConsecutiveIntDeviceArray(int size);
    SharedCudaPtr<int> GenerateRandomStencil(int size);

private:
    curandGenerator_t gen;
};

} /* namespace ddj */
#endif /* DDJ_HELPER_CUDA_H_ */
