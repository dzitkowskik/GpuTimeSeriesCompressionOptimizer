#include "helper_generator.h"
#include "helper_macros.h"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>

namespace ddj
{

    float* HelperGenerator::GenerateRandomDeviceArray(int n)
    {
        float* d_result;
        curandGenerator_t gen;
        CUDA_CALL(cudaMalloc((void**)&d_result, n * sizeof(float)));
        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1991ULL));
        CURAND_CALL(curandGenerateUniform(gen, d_result, n));
        CURAND_CALL(curandDestroyGenerator(gen));
        return d_result;
    }

} /* nemespace ddj */
