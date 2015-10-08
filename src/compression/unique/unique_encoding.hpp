/*
 * unique_encoding.hpp
 *
 *  Created on: 8 paź 2015
 *      Author: ghash
 */

#ifndef UNIQUE_ENCODING_HPP_
#define UNIQUE_ENCODING_HPP_

#include "core/cuda_ptr.hpp"
#include "core/execution_policy.hpp"

#include <gtest/gtest.h>

namespace ddj {

class UniqueEncoding
{
public:

private:
	template<typename T>
	SharedCudaPtr<char> CompressUnique(SharedCudaPtr<T> data, SharedCudaPtr<T> unique);

	template<typename T>
	SharedCudaPtr<T> DecompressUnique(SharedCudaPtr<char> data);

private:
	ExecutionPolicy _policy;

	friend class DictEncoding;
	friend class MostFrequentTest;
	FRIEND_TEST(MostFrequentTest, CompressDecompressMostFrequent_random_int);
};

} /* namespace ddj */

#endif /* UNIQUE_ENCODING_HPP_ */
