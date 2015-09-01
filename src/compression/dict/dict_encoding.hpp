/*
 *  dict_encoding.hpp
 *
 *  Created on: 18-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_DICT_ENCODING_HPP_
#define DDJ_DICT_ENCODING_HPP_

#include "core/cuda_ptr.hpp"
#include "core/execution_policy.hpp"
#include "helpers/helper_cudakernels.cuh"

namespace ddj {

class DictEncoding
{
	ExecutionPolicy _policy;

public:
	// TODO: Implement as templates (DictEncoding)
	// template<typename T> SharedCudaPtrVector<char>
	// Encode(SharedCudaPtr<T> data);
	//
	// template<typename T> SharedCudaPtr<T>
	// Decode(SharedCudaPtr<char> dataMostFrequent, SharedCudaPtr<char> dataRest);

	// ONLY FOR NOW
	SharedCudaPtrVector<char> Encode(SharedCudaPtr<int> data);
	// SharedCudaPtr<T> Decode(SharedCudaPtr<char> dataMostFrequent, SharedCudaPtr<char> dataRest);

private:
    SharedCudaPtr<int> GetMostFrequentStencil(
		SharedCudaPtr<int> data, SharedCudaPtr<int> mostFrequent);

    SharedCudaPtr<char> CompressMostFrequent(
        SharedCudaPtr<int> data, SharedCudaPtr<int> mostFrequent);

	SharedCudaPtr<int> DecompressMostFrequent(
		SharedCudaPtr<char> data);

	HelperCudaKernels _cudaKernels;
};

} /* namespace ddj */
#endif /* DDJ_DICT_ENCODING_HPP_ */
