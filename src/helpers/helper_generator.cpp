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

	SharedCudaPtr<int> HelperGenerator::GenerateRandomStencil(int n)
	{
		HelperCudaKernels kernels;
		auto result = this->GenerateRandomIntDeviceArray(n);
		kernels.ModuloInPlaceKernel(result, 2);
		return result;
	}

	SharedCudaPtr<int> HelperGenerator::GenerateRandomIntDeviceArray(int n, int from, int to)
	{
		if(to <= from)
			throw std::runtime_error("HelperGenerator - from must be less than to");
		int distance = to - from;
		HelperCudaKernels kernels;
		auto result = CudaPtr<int>::make_shared(n);
		CURAND_CALL(curandGenerate(gen, (unsigned int*)result->get(), n));
		kernels.ModuloInPlaceKernel(result, distance);
		kernels.AdditionInPlaceKernel(result, from);
		return result;
	}

} /* nemespace ddj */
