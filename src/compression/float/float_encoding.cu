/*
 *  float_encoding.cu
 *
 *  Created on: 30 paź 2015
 *      Author: Karol Dzitkowski
 */

#include <compression/float/float_encoding.hpp>
#include <util/transform/cuda_array_transform.hpp>
#include <util/statistics/cuda_array_statistics.hpp>
#include "core/macros.h"

namespace ddj
{
	template<typename T>
	SharedCudaPtrVector<char> FloatEncoding::Encode(SharedCudaPtr<T> data)
	{
		int precision = CudaArrayStatistics().Precision(data);

		FloatingPointToIntegerOperator<T, int> op { precision };
		auto resultData = CudaArrayTransform().Transform<T, int>(data, op);
		auto resultMetadata = CudaPtr<char>::make_shared(sizeof(int));
		resultMetadata->fillFromHost((char*)&precision, sizeof(int));

		return SharedCudaPtrVector<char> { resultMetadata, MoveSharedCudaPtr<int, char>(resultData) };
	}

	template<typename T>
	SharedCudaPtr<T> FloatEncoding::Decode(SharedCudaPtrVector<char> input)
	{
		auto metadata = input[0];
		auto data = MoveSharedCudaPtr<char, int>(input[1]);

		int precision;
		CUDA_CALL( cudaMemcpy(&precision, metadata->get(), sizeof(int), CPY_DTH) );

		IntegerToFloatingPointOperator<int, T> op { precision };
		auto result = CudaArrayTransform().Transform<int, T>(data, op);
		return result;
	}

#define FLOAT_ENCODING_SPEC(X) \
	template SharedCudaPtrVector<char> FloatEncoding::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> FloatEncoding::Decode<X>(SharedCudaPtrVector<char> data);
FOR_EACH(FLOAT_ENCODING_SPEC, float, int, long long, unsigned int)

} /* namespace ddj */
