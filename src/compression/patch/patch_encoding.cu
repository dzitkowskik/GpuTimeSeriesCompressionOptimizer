/*
 *  patched_data.cu
 *
 *  Created on: 13-05-2015
 *      Author: Karol Dzitkowski
 */

#include "compression/patch/patch_encoding.hpp"
#include "helpers/helper_cuda.cuh"
#include "helpers/helper_print.hpp"
#include "core/macros.h"
#include "util/stencil/stencil.hpp"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

namespace ddj {

template<typename UnaryOperator>
template<typename T>
SharedCudaPtrVector<char> PatchEncoding<UnaryOperator>::Encode(SharedCudaPtr<T> data)
{
	if(data->size() <= 0)
		return SharedCudaPtrVector<char>{ CudaPtr<char>::make_shared(), CudaPtr<char>::make_shared() };

    int size = data->size();

    // Create stencil
    Stencil stencil = Stencil::Create(data, this->_op);

    // Compress the stencil
    auto stencilPacked = stencil.pack();

    // Split according to the stencil
    auto splittedData = this->_splitter.Split(data, *stencil);

    // Return results as vector with compressed stencil as metadata
    auto operatorTrue = MoveSharedCudaPtr<T, char>(std::get<0>(splittedData));
    auto operatorFalse = MoveSharedCudaPtr<T, char>(std::get<1>(splittedData));

//    printf("patch split %d | %d\n", operatorTrue->size(), operatorFalse->size());

    return SharedCudaPtrVector<char> {stencilPacked, operatorTrue, operatorFalse};
}

template<typename UnaryOperator>
template<typename T>
SharedCudaPtr<T> PatchEncoding<UnaryOperator>::Decode(SharedCudaPtrVector<char> data)
{
	if(data[1]->size() <= 0 && data[2]->size() <= 0)
		return CudaPtr<T>::make_shared();

	auto stencilMetadata = data[0];
	auto operatorTrue = MoveSharedCudaPtr<char, T>(data[1]);
	auto operatorFalse = MoveSharedCudaPtr<char, T>(data[2]);

	// Uncompress stencil
	auto stencil = Stencil().unpack(stencilMetadata);

	return this->_splitter.Merge(std::make_tuple(operatorTrue, operatorFalse), stencil);
}

template<typename UnaryOperator>
SharedCudaPtrVector<char> PatchEncoding<UnaryOperator>::EncodeInt(SharedCudaPtr<int> data)
{
	return this->Encode<int>(data);
}

template<typename UnaryOperator>
SharedCudaPtr<int> PatchEncoding<UnaryOperator>::DecodeInt(SharedCudaPtrVector<char> data)
{
	return this->Decode<int>(data);
}

template<typename UnaryOperator>
SharedCudaPtrVector<char> PatchEncoding<UnaryOperator>::EncodeFloat(SharedCudaPtr<float> data)
{
	return this->Encode<float>(data);
}

template<typename UnaryOperator>
SharedCudaPtr<float> PatchEncoding<UnaryOperator>::DecodeFloat(SharedCudaPtrVector<char> data)
{
	return this->Decode<float>(data);
}

#define PATCH_ENCODING_OPERATORS_SPEC(X) \
	template SharedCudaPtrVector<char> PatchEncoding<OutsideOperator<X>>::EncodeInt(SharedCudaPtr<int> data); \
	template SharedCudaPtrVector<char> PatchEncoding<OutsideOperator<X>>::EncodeFloat(SharedCudaPtr<float> data); \
	template SharedCudaPtr<int> PatchEncoding<OutsideOperator<X>>::DecodeInt(SharedCudaPtrVector<char> data); \
	template SharedCudaPtr<float> PatchEncoding<OutsideOperator<X>>::DecodeFloat(SharedCudaPtrVector<char> data); \
	\
	template SharedCudaPtrVector<char> PatchEncoding<LowerOperator<X>>::EncodeInt(SharedCudaPtr<int> data); \
	template SharedCudaPtrVector<char> PatchEncoding<LowerOperator<X>>::EncodeFloat(SharedCudaPtr<float> data); \
	template SharedCudaPtr<int> PatchEncoding<LowerOperator<X>>::DecodeInt(SharedCudaPtrVector<char> data); \
	template SharedCudaPtr<float> PatchEncoding<LowerOperator<X>>::DecodeFloat(SharedCudaPtrVector<char> data);
FOR_EACH(PATCH_ENCODING_OPERATORS_SPEC, float, int, long long, unsigned int)

#define PATCH_ENCODING_SPEC(X) \
	template SharedCudaPtrVector<char> PatchEncoding<OutsideOperator<X>>::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> PatchEncoding<OutsideOperator<X>>::Decode<X>(SharedCudaPtrVector<char> data); \
	\
	template SharedCudaPtrVector<char> PatchEncoding<LowerOperator<X>>::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> PatchEncoding<LowerOperator<X>>::Decode<X>(SharedCudaPtrVector<char> data);
FOR_EACH(PATCH_ENCODING_SPEC, float, int, long long, unsigned int)


} /* namespace ddj */
