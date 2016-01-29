/*
 * outside_patch_encoding.cpp
 *
 *  Created on: Nov 14, 2015
 *      Author: Karol Dzitkowski
 */

#include "compression/patch/outside_patch_encoding.hpp"

#include "core/cuda_launcher.cuh"

#include "core/macros.h"
#include "util/stencil/stencil.hpp"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

namespace ddj
{

template<typename T>
OutsideOperator<T> OutsidePatchEncoding::GetOperator()
{
	T dist = _max - _min;
	OutsideOperator<T> op
	{
		static_cast<T>(_min + _factor * dist),
		static_cast<T>(_max - _factor * dist)
	};
    return op;
}

template<typename T>
SharedCudaPtrVector<char> OutsidePatchEncoding::Encode(SharedCudaPtr<T> data)
{
	if(data->size() <= 0)
		return SharedCudaPtrVector<char>{
			CudaPtr<char>::make_shared(),
			CudaPtr<char>::make_shared(),
			CudaPtr<char>::make_shared()};

    int size = data->size();

    // Split according to the stencil
    Stencil stencil = Stencil::Create(data, GetOperator<T>());
    auto stencilPacked = stencil.pack();
    auto splittedData = this->_splitter.Split(data, *stencil);

    // Return results as vector with compressed stencil as metadata
    auto operatorTrue = MoveSharedCudaPtr<T, char>(std::get<0>(splittedData));
    auto operatorFalse = MoveSharedCudaPtr<T, char>(std::get<1>(splittedData));

    return SharedCudaPtrVector<char> {stencilPacked, operatorTrue, operatorFalse};
}

template<typename T>
SharedCudaPtr<T> OutsidePatchEncoding::Decode(SharedCudaPtrVector<char> data)
{
	if(data[1]->size() <= 0 && data[2]->size() <= 0)
		return CudaPtr<T>::make_shared();

	auto stencilMetadata = data[0];
	auto operatorTrue = CastSharedCudaPtr<char, T>(data[1]);
	auto operatorFalse = CastSharedCudaPtr<char, T>(data[2]);

	// Uncompress stencil
	auto stencil = Stencil().unpack(stencilMetadata);

	return this->_splitter.Merge(std::make_tuple(operatorTrue, operatorFalse), stencil);
}

#define OUTSIDE_PATCH_ENCODING_SPEC(X) \
	template SharedCudaPtrVector<char> OutsidePatchEncoding::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> OutsidePatchEncoding::Decode<X>(SharedCudaPtrVector<char> data);
FOR_EACH(OUTSIDE_PATCH_ENCODING_SPEC, char, short, double, float, int, long, long long, unsigned int)


} /* namespace ddj */
