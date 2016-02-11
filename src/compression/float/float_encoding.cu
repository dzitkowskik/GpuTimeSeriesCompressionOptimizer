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
#include <limits>
#include <cmath>

namespace ddj
{
	template<typename T>
	SharedCudaPtrVector<char> FloatEncoding::Encode(SharedCudaPtr<T> data)
	{
		CUDA_ASSERT_RETURN( cudaGetLastError() );
	    LOG4CPLUS_INFO_FMT(_logger, "FLOAT encoding START: data size = %lu", data->size());

		if(data->size() <= 0)
			return SharedCudaPtrVector<char>{
						CudaPtr<char>::make_shared(),
						CudaPtr<char>::make_shared()
						};

		int precision = CudaArrayStatistics().Precision(data);
		SharedCudaPtr<int> resultData;
		FloatingPointToIntegerOperator<T, int> op { precision };

		// Make sure we won't overflow
		bool transform = false;
		if(precision < MAX_PRECISION)
		{
			auto minMax = CudaArrayStatistics().MinMax(data);
			int scaleFactor = std::pow(10, precision);
			if((std::get<0>(minMax) * scaleFactor) > std::numeric_limits<int>::min() &&
					(std::get<1>(minMax) * scaleFactor) < std::numeric_limits<int>::max())
				transform = true;
			else precision = MAX_PRECISION;
		}

		if(transform)
			resultData = CudaArrayTransform().Transform<T, int>(data, op);
		else
			resultData = CastSharedCudaPtr<T, int>(data->copy());

		auto resultMetadata = CudaPtr<char>::make_shared(sizeof(int));
		resultMetadata->fillFromHost((char*)&precision, sizeof(int));

		CUDA_ASSERT_RETURN( cudaGetLastError() );
	    LOG4CPLUS_INFO(_logger, "FLOAT enoding END");

		return SharedCudaPtrVector<char> { resultMetadata, MoveSharedCudaPtr<int, char>(resultData) };
	}

	template<typename T>
	SharedCudaPtr<T> FloatEncoding::Decode(SharedCudaPtrVector<char> input)
	{
		LOG4CPLUS_INFO_FMT(
			_logger,
			"FLOAT decoding START: input[0] size = %lu, input[1] size = %lu",
			input[0]->size(), input[1]->size()
		);

		if(input[1]->size() <= 0)
			return CudaPtr<T>::make_shared();

		auto metadata = input[0];
		auto data = MoveSharedCudaPtr<char, int>(input[1]);

		int precision;
		CUDA_CALL( cudaMemcpy(&precision, metadata->get(), sizeof(int), CPY_DTH) );

		SharedCudaPtr<T> result;
		IntegerToFloatingPointOperator<int, T> op { precision };
		if(precision < MAX_PRECISION)
			result = CudaArrayTransform().Transform<int, T>(data, op);
		else
			result = CastSharedCudaPtr<int, T>(data->copy());

		CUDA_ASSERT_RETURN( cudaGetLastError() );
	    LOG4CPLUS_INFO(_logger, "FLOAT decoding END");

		return result;
	}

#define FLOAT_ENCODING_SPEC(X) \
	template SharedCudaPtrVector<char> FloatEncoding::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> FloatEncoding::Decode<X>(SharedCudaPtrVector<char> data);
FOR_EACH(FLOAT_ENCODING_SPEC, char, short, double, float, int, long, long long, unsigned int)

} /* namespace ddj */
