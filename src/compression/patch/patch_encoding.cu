/*
 *  patched_data.cu
 *
 *  Created on: 13-05-2015
 *      Author: Karol Dzitkowski
 */

#include "compression/patch/patch_encoding.hpp"
#include "helpers/helper_cuda.cuh"
#include "core/macros.h"
#include "core/operators.cuh"
#include "util/stencil/stencil.hpp"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

namespace ddj {

template<typename T, typename UnaryOperator>
__global__ void _stencilKernel(T* data, int size, int* out, UnaryOperator op)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= size) return;
    T value = data[idx];
    out[idx] = op(value) ? 1 : 0;
}

template<typename UnaryOperator>
template<typename T>
SharedCudaPtrVector<char> PatchEncoding<UnaryOperator>::Encode(SharedCudaPtr<T> data)
{
    int size = data->size();

    // Create stencil
    Stencil stencil = Stencil(CudaPtr<int>::make_shared(size));
    this->_policy.setSize(size);
    cudaLaunch(this->_policy, _stencilKernel<T, UnaryOperator>,
        data->get(),
        data->size(),
        stencil->get(),
        this->_op);

    // Compress the stencil
    auto stencilPacked = stencil.pack();

    // Split according to the stencil
    auto splittedData = this->_splitter.Split(data, *stencil);

    // Return results as vector with compressed stencil as metadata
    auto operatorTrue = MoveSharedCudaPtr<T, char>(std::get<0>(splittedData));
    auto operatorFalse = MoveSharedCudaPtr<T, char>(std::get<1>(splittedData));
    return SharedCudaPtrVector<char> {stencilPacked, operatorTrue, operatorFalse};
}

template<typename UnaryOperator>
template<typename T>
SharedCudaPtr<T> PatchEncoding<UnaryOperator>::Decode(SharedCudaPtrVector<char> data)
{
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


#define PATCH_ENCODING_SPEC(X) \
	template SharedCudaPtrVector<char> PatchEncoding<OutsideOperator<X,X>>::Encode<X>(SharedCudaPtr<X> data); \
	template SharedCudaPtr<X> PatchEncoding<OutsideOperator<X,X>>::Decode<X>(SharedCudaPtrVector<char> data); \
	template SharedCudaPtrVector<char> PatchEncoding<OutsideOperator<X,X>>::EncodeInt(SharedCudaPtr<int> data); \
	template SharedCudaPtr<int> PatchEncoding<OutsideOperator<X,X>>::DecodeInt(SharedCudaPtrVector<char> data); \
	template SharedCudaPtrVector<char> PatchEncoding<OutsideOperator<X,X>>::EncodeFloat(SharedCudaPtr<float>); \
	template SharedCudaPtr<float> PatchEncoding<OutsideOperator<X,X>>::DecodeFloat(SharedCudaPtrVector<char>);
FOR_EACH(PATCH_ENCODING_SPEC, float, int, long long, unsigned int)

} /* namespace ddj */
