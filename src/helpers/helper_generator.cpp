#include "helper_generator.hpp"
#include "helper_macros.h"
#include "helper_cudakernels.cuh"
#include "core/cuda_ptr.hpp"

#include <time.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

namespace ddj
{
	HelperGenerator::HelperGenerator()
    {
		CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1991ULL ^ time(NULL)));
    }

	HelperGenerator::~HelperGenerator()
	{
		CURAND_CALL(curandDestroyGenerator(gen));
	}

	SharedCudaPtr<float> HelperGenerator::GenerateRandomFloatDeviceArray(int n)
    {
        auto result = CudaPtr<float>::make_shared(n);
        CURAND_CALL(curandGenerateUniform(gen, result->get(), n));
        return result;
    }

	SharedCudaPtr<int> HelperGenerator::GenerateRandomIntDeviceArray(int n)
	{
		auto result = CudaPtr<int>::make_shared(n);
		CURAND_CALL(curandGenerate(gen, (unsigned int*)result->get(), n));
		return result;
	}

	SharedCudaPtr<int> HelperGenerator::GenerateConsecutiveIntDeviceArray(int size)
	{
		HelperCudaKernels kernels;
		return kernels.CreateConsecutiveNumbersArray<int>(size, 0);
	}

} /* nemespace ddj */
