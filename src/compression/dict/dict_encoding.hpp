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

namespace ddj {

class DictEncoding
{
	ExecutionPolicy _policy;

public:
	template<typename T> SharedCudaPtr<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtr<char> data);

private:
    SharedCudaPtr<int> GetMostFrequentStencil(
        SharedCudaPtr<int> data,
        SharedCudaPtr<int> mostFrequent);

    SharedCudaPtr<char> CompressMostFrequent(
        SharedCudaPtr<int> data, SharedCudaPtr<int> mostFrequent);
};

} /* namespace ddj */
#endif /* DDJ_DICT_ENCODING_HPP_ */
