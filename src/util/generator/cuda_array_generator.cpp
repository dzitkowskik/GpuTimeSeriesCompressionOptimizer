#include "util/generator/cuda_array_generator.hpp"
#include "helpers/helper_macros.h"
#include "helpers/helper_cudakernels.cuh"
#include "core/cuda_ptr.hpp"

#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

namespace ddj
{
	CudaArrayGenerator::CudaArrayGenerator()
    {
		CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1991ULL ^ time(NULL)));
    }

	CudaArrayGenerator::~CudaArrayGenerator()
	{
		CURAND_CALL(curandDestroyGenerator(gen));
	}

	SharedCudaPtr<float> CudaArrayGenerator::GenerateRandomFloatDeviceArray(int n)
    {
        auto result = CudaPtr<float>::make_shared(n);
        CURAND_CALL(curandGenerateUniform(gen, result->get(), n));
        return result;
    }

	SharedCudaPtr<int> CudaArrayGenerator::GenerateRandomIntDeviceArray(int n)
	{
		auto result = CudaPtr<int>::make_shared(n);
		CURAND_CALL(curandGenerate(gen, (unsigned int*)result->get(), n));
		return result;
	}

	SharedCudaPtr<int> CudaArrayGenerator::GenerateConsecutiveIntDeviceArray(int size)
	{
		HelperCudaKernels kernels;
		return kernels.CreateConsecutiveNumbersArray<int>(size, 0);
	}

	SharedCudaPtr<int> CudaArrayGenerator::GenerateRandomStencil(int n)
	{
		HelperCudaKernels kernels;
		auto result = this->GenerateRandomIntDeviceArray(n);
		kernels.AbsoluteInPlaceKernel(result);
		kernels.ModuloInPlaceKernel(result, 2);
		return result;
	}

	SharedCudaPtr<int> CudaArrayGenerator::GenerateRandomIntDeviceArray(int n, int from, int to)
	{
		int distance = std::abs(to - from);
		HelperCudaKernels kernels;
		auto result = CudaPtr<int>::make_shared(n);
		CURAND_CALL(curandGenerate(gen, (unsigned int*)result->get(), n));
		kernels.AbsoluteInPlaceKernel(result);
		kernels.ModuloInPlaceKernel(result, distance);
		kernels.AdditionInPlaceKernel(result, from);
		return result;
	}

} /* nemespace ddj */
