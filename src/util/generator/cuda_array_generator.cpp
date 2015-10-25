#include "util/generator/cuda_array_generator.hpp"
#include "helpers/helper_macros.h"
#include "core/cuda_ptr.hpp"
#include "core/operators.cuh"

#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

namespace ddj
{
	CudaArrayGenerator::CudaArrayGenerator()
    {
		CURAND_CALL(curandCreateGenerator(&this->_gen, CURAND_RNG_PSEUDO_DEFAULT));
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(this->_gen, 1991ULL ^ time(NULL)));
    }

	CudaArrayGenerator::~CudaArrayGenerator()
	{
		CURAND_CALL(curandDestroyGenerator(this->_gen));
	}

	SharedCudaPtr<float> CudaArrayGenerator::GenerateRandomFloatDeviceArray(int n)
    {
        auto result = CudaPtr<float>::make_shared(n);
        CURAND_CALL(curandGenerateUniform(this->_gen, result->get(), n));
        return result;
    }

	SharedCudaPtr<int> CudaArrayGenerator::GenerateRandomIntDeviceArray(int n)
	{
		auto result = CudaPtr<int>::make_shared(n);
		CURAND_CALL(curandGenerate(this->_gen, (unsigned int*)result->get(), n));
		return result;
	}

	SharedCudaPtr<int> CudaArrayGenerator::GenerateConsecutiveIntDeviceArray(int size)
	{
		return CreateConsecutiveNumbersArray<int>(size, 0);
	}

	SharedCudaPtr<int> CudaArrayGenerator::GenerateRandomStencil(int n)
	{
		auto result = this->GenerateRandomIntDeviceArray(n);
		this->_transform.TransformInPlace(result, AbsoluteOperator<int>());
		this->_transform.TransformInPlace(result, ModulusOperator<int> {2});
		return result;
	}

	SharedCudaPtr<int> CudaArrayGenerator::GenerateRandomIntDeviceArray(int n, int from, int to)
	{
		int distance = std::abs(to - from);
		auto result = CudaPtr<int>::make_shared(n);
		CURAND_CALL(curandGenerate(this->_gen, (unsigned int*)result->get(), n));
		this->_transform.TransformInPlace(result, AbsoluteOperator<int>());
		this->_transform.TransformInPlace(result, ModulusOperator<int> {distance});
		this->_transform.TransformInPlace(result, AdditionOperator<int> {from});
		return result;
	}

} /* nemespace ddj */
