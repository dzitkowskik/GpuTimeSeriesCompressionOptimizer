/*
 * helper_generator.h 26-03-2015 Karol Dzitkowski
 */

#ifndef DDJ_HELPER_GENERATOR_H_
#define DDJ_HELPER_GENERATOR_H_

#include "../core/logger.h"
#include "../core/config.h"
#include <boost/noncopyable.hpp>
#include <curand.h>

namespace ddj {

class HelperGenerator : private boost::noncopyable
{
private:
    /* LOGGER & CONFIG */
    Logger _logger;
    Config* _config;

public:
    float* GenerateRandomFloatDeviceArray(int size);
    int* GenerateRandomIntDeviceArray(int size);
    curandGenerator_t gen;

public:
    HelperGenerator();
    ~HelperGenerator();
};

} /* namespace ddj */
#endif /* DDJ_HELPER_CUDA_H_ */
