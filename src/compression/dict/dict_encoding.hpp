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
public:
	DictEncoding() : _freqCnt(4) {}
	~DictEncoding() {}
	DictEncoding(const DictEncoding&) = default;
	DictEncoding(DictEncoding&&) = default;

public:
	template<typename T> SharedCudaPtrVector<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtrVector<char> data);

private:
	template<typename T>
    SharedCudaPtr<int> GetMostFrequentStencil(SharedCudaPtr<T> data, SharedCudaPtr<T> mostFrequent);

public:
	void SetFreqCnt(int freqCnt) { _freqCnt = freqCnt; }

private:
	int _freqCnt;

private:
	Splitter _splitter;
	ExecutionPolicy _policy;

	friend class DictCompressionTest;
	FRIEND_TEST(DictCompressionTest, CompressDecompressMostFrequent_random_int);
};

} /* namespace ddj */
#endif /* DDJ_DICT_ENCODING_HPP_ */
