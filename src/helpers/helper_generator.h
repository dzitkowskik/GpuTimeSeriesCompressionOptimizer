/*
 * helper_generator.h 26-03-2015 Karol Dzitkowski
 */

#ifndef DDJ_HELPER_GENERATOR_H_
#define DDJ_HELPER_GENERATOR_H_

#include "../core/logger.h"
#include "../core/config.h"

namespace ddj {

class HelperGenerator {
private:
    /* LOGGER & CONFIG */
    Logger _logger;
    Config* _config;

public:
    HelperGenerator()
        : _logger(Logger::getRoot()), _config(Config::GetInstance()) { }

    float* GenerateRandomDeviceArray(int size);
};

} /* namespace ddj */
#endif /* DDJ_HELPER_CUDA_H_ */
