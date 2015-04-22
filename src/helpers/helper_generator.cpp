#include "helper_generator.h"
#include "helper_macros.h"

#include <cuda_runtime_api.h>
#include <cuda.h>

namespace ddj
{
	HelperGenerator::HelperGenerator()
    {
		CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1991ULL));
    }

	HelperGenerator::~HelperGenerator()
	{
		CURAND_CALL(curandDestroyGenerator(gen));
	}

    float* HelperGenerator::GenerateRandomFloatDeviceArray(int n)
    {
        float* d_result;
        CUDA_CALL(cudaMalloc((void**)&d_result, n * sizeof(float)));
        CURAND_CALL(curandGenerateUniform(gen, d_result, n));
        return d_result;
    }

    int* HelperGenerator::GenerateRandomIntDeviceArray(int n)
	{
		unsigned int* d_result;
		CUDA_CALL(cudaMalloc((void**)&d_result, n * sizeof(int)));
		CURAND_CALL(curandGenerate(gen, d_result, n));
		return (int*)d_result;
	}

} /* nemespace ddj */
