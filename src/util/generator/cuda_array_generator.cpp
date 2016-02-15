#include "util/generator/cuda_array_generator.hpp"
#include "core/macros.h"
#include "core/cuda_ptr.hpp"

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

	SharedCudaPtr<double> CudaArrayGenerator::GenerateRandomDoubleDeviceArray(int n)
    {
        auto result = CudaPtr<double>::make_shared(n);
        CURAND_CALL(curandGenerateUniformDouble(this->_gen, result->get(), n));
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
		this->_transform.TransformInPlace(result, AbsoluteOperator<int, int>());
		this->_transform.TransformInPlace(result, ModulusOperator<int, int> {2});
		return result;
	}

	SharedCudaPtr<int> CudaArrayGenerator::GenerateRandomIntDeviceArray(int n, int from, int to)
	{
		int distance = std::abs(to - from);
		auto result = CudaPtr<int>::make_shared(n);
		CURAND_CALL(curandGenerate(this->_gen, (unsigned int*)result->get(), n));
		this->_transform.TransformInPlace(result, AbsoluteOperator<int, int>());
		this->_transform.TransformInPlace(result, ModulusOperator<int, int> {distance});
		this->_transform.TransformInPlace(result, AdditionOperator<int, int> {from});
		return result;
	}

	template<typename T>
	SharedCudaPtr<T> CudaArrayGenerator::GetFakeDataWithPatternA(
			int part,
			size_t len,
			T step,
			T min,
			T max,
			size_t size)
	{
		T oldMin = min;
		auto h_result = new T[size];
		size_t start = part*size;

		// Prepare data
		min = min + (T)(start/len)*step;
		for(size_t i = 0; i < size; i++)
		{
			auto value = min < max ? (min > oldMin ? min : oldMin) : max;
			if((start+i) % len == 0)
			{
				value = max;
				min += step;
				if(min > max || min < oldMin) step = -step;
			}
			h_result[i] = value;
		}

		auto d_result = CudaPtr<T>::make_shared(size);
		d_result->fillFromHost(h_result, size);
		delete [] h_result;
		return d_result;
	}

	template<typename T>
	SharedCudaPtr<T> CudaArrayGenerator::GetFakeDataWithPatternB(
			int part,
			size_t len,
			T min,
			T max,
			size_t size)
	{
		int maxRand = 5;
		srand(time(NULL));
		auto h_result = new T[size];
		size_t start = part*size;
		auto value = max;
		auto step = (max-min)/(len/2);

		// Prepare data
		for(size_t i = 0; i < size; i++)
		{
			if((start+i)/len % 2 == 0)	// pattern1
			{
				auto randomInt = rand() % (maxRand+1);
				switch(i%2)
				{
					case 0: value = max-randomInt; break;
					case 1: value = min+randomInt; break;
				}
			} else { // pattern2
				auto x = (start+i) % len;
				value = x < (len/2) ? max - x*step : max-(len-x)*step;
			}
			h_result[i] = value;
		}

		auto d_result = CudaPtr<T>::make_shared(size);
		d_result->fillFromHost(h_result, size);
		delete [] h_result;
		return d_result;
	}

	SharedCudaPtr<time_t> CudaArrayGenerator::GetFakeDataForTime(
			time_t min,
			double flatness,
			size_t size)
	{
		int maxStep=2;
		srand(time(NULL));
		auto h_result = new time_t[size];
		auto value = min;

		// Prepare data
		for(size_t i = 0; i < size; i++)
		{
			auto step = 1 + (rand() % maxStep);
			if(rand()%100 > flatness*100) value += step;
			h_result[i] = value;
		}

		auto d_result = CudaPtr<time_t>::make_shared(size);
		d_result->fillFromHost(h_result, size);
		delete [] h_result;
		return d_result;
	}

	#define GENERATOR_SPEC(X) \
		template SharedCudaPtr<X> CudaArrayGenerator::GetFakeDataWithPatternA<X>(int, size_t, X, X, X, size_t); \
		template SharedCudaPtr<X> CudaArrayGenerator::GetFakeDataWithPatternB<X>(int, size_t, X, X, size_t);
	FOR_EACH(GENERATOR_SPEC, char, short, float, int, long, long long, unsigned int)


} /* nemespace ddj */
