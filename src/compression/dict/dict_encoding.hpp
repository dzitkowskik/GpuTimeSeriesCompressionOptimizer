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
#include "util/splitter/splitter.hpp"
#include <gtest/gtest.h>

namespace ddj {

class DictEncoding
{
	ExecutionPolicy _policy;

public:
	// TODO: Implement as templates
	SharedCudaPtrVector<char> Encode(SharedCudaPtr<int> data);
	SharedCudaPtr<int> Decode(SharedCudaPtrVector<char> data);

private:
	SharedCudaPtr<int> GetMostFrequent(
		SharedCudaPtrPair<int, int> histogram, int freqCnt);

    SharedCudaPtr<int> GetMostFrequentStencil(
		SharedCudaPtr<int> data, SharedCudaPtr<int> mostFrequent);

    SharedCudaPtr<char> CompressMostFrequent(
        SharedCudaPtr<int> data, SharedCudaPtr<int> mostFrequent);

	SharedCudaPtr<int> DecompressMostFrequent(SharedCudaPtr<char> data, int freqCnt, int outputSize);

	friend class DictCompressionTest;
 	FRIEND_TEST(DictCompressionTest, GetMostFrequent_fake_data);
	FRIEND_TEST(DictCompressionTest, GetMostFrequent_random_int);
	FRIEND_TEST(DictCompressionTest, CompressDecompressMostFrequent_random_int);

private:
	Splitter _splitter;
};

} /* namespace ddj */
#endif /* DDJ_DICT_ENCODING_HPP_ */
