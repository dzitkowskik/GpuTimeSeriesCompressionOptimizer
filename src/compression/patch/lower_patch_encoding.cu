/*
 * lower_patch_encoding.cpp
 *
 *  Created on: Nov 14, 2015
 *      Author: Karol Dzitkowski
 */

#include "compression/patch/lower_patch_encoding.hpp"

#include "core/cuda_launcher.cuh"
#include "core/cuda_array.hpp"
#include "core/macros.h"
#include "util/stencil/stencil.hpp"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

namespace ddj
{

template<typename T>
LowerOperator<T> LowerPatchEncoding::GetOperator()
{
	T dist = _max - _min;
	LowerOperator<T> op
	{
		static_cast<T>(_max - _factor * dist)
	};
    return op;
}

template<typename T>
SharedCudaPtrVector<char> LowerPatchEncoding::Encode(SharedCudaPtr<T> data)
{
	CUDA_ASSERT_RETURN( cudaGetLastError() );
    LOG4CPLUS_INFO_FMT(_logger, "PATCH (LOWER) encoding START: data size = %lu", data->size());

	if(data->size() <= 0)
		return SharedCudaPtrVector<char>{
			CudaPtr<char>::make_shared(),
			CudaPtr<char>::make_shared(),
			CudaPtr<char>::make_shared()};

    int size = data->size();

    // Split according to the stencil
    auto op = GetOperator<T>();

    LOG4CPLUS_TRACE(_logger, "PATCH (LOWER) - OP VALUE = " << op.value);

    Stencil stencil = Stencil::Create(data, op);
    auto stencilPacked = stencil.pack();
    auto splittedData = this->_splitter.Split(data, *stencil);

    LOG4CPLUS_TRACE_FMT(_logger, "LEFT: %s ...", CudaArray().ToString(std::get<0>(splittedData)->copy()).c_str());
    LOG4CPLUS_TRACE_FMT(_logger, "RIGHT: %s ...", CudaArray().ToString(std::get<1>(splittedData)->copy()).c_str());

    // Return results as vector with compressed stencil as metadata
    auto operatorTrue = CastSharedCudaPtr<T, char>(std::get<0>(splittedData));
    auto operatorFalse = CastSharedCudaPtr<T, char>(std::get<1>(splittedData));

	CUDA_ASSERT_RETURN( cudaGetLastError() );
    LOG4CPLUS_INFO(_logger, "PATCH (LOWER) enoding END");

    return SharedCudaPtrVector<char> {stencilPacked, operatorTrue, operatorFalse};
}

template<typename T>
SharedCudaPtr<T> LowerPatchEncoding::Decode(SharedCudaPtrVector<char> input)
{
	LOG4CPLUS_INFO_FMT(
		_logger,
		"PATCH (LOWER) decoding START: input[0] size = %lu, input[1] size = %lu, input[2] size = %lu",
		input[0]->size(), input[1]->size(), input[2]->size()
	);

	if(input[1]->size() <= 0 && input[2]->size() <= 0)
		return CudaPtr<T>::make_shared();

	auto stencilMetadata = input[0];
	auto operatorTrue = CastSharedCudaPtr<char, T>(input[1]);
	auto operatorFalse = CastSharedCudaPtr<char, T>(input[2]);

	// Uncompress stencil
	auto stencil = Stencil().unpack(stencilMetadata);

	auto result = this->_splitter.Merge(std::make_tuple(operatorTrue, operatorFalse), stencil);

	CUDA_ASSERT_RETURN( cudaGetLastError() );
    LOG4CPLUS_INFO(_logger, "PATCH (LOWER) decoding END");

	return result;
}

#define OUTSIDE_PATCH_ENCODING_SPEC(X) \
	template SharedCudaPtrVector<char> LowerPatchEncoding::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> LowerPatchEncoding::Decode<X>(SharedCudaPtrVector<char> data);
FOR_EACH(OUTSIDE_PATCH_ENCODING_SPEC, char, short, double, float, int, long, long long, unsigned int)


} /* namespace ddj */
