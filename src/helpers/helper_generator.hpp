/*
 * helper_generator.hpp 26-03-2015 Karol Dzitkowski
 */

#ifndef DDJ_HELPER_GENERATOR_H_
#define DDJ_HELPER_GENERATOR_H_

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
    float* GenerateRandomFloatDeviceArray(int size);
    int* GenerateRandomIntDeviceArray(int size);

    template<typename T>
    T* GenerateRandomDeviceArray(int size);

    curandGenerator_t gen;
};

} /* namespace ddj */
#endif /* DDJ_HELPER_CUDA_H_ */
