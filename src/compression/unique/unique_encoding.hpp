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
#include "compression/encoding.hpp"
#include <gtest/gtest.h>

namespace ddj {

class UniqueEncoding : public Encoding
{
public:
	UniqueEncoding(){}
	~UniqueEncoding(){}
	UniqueEncoding(const UniqueEncoding&) = default;
	UniqueEncoding(UniqueEncoding&&) = default;

public:
	unsigned int GetNumberOfResults() { return 1; }

protected:
	SharedCudaPtrVector<char> EncodeInt(SharedCudaPtr<int> data)
	{
		return this->Encode<int>(data);
	}

	SharedCudaPtr<int> DecodeInt(SharedCudaPtrVector<char> data)
	{
		return this->Decode<int>(data);
	}

	SharedCudaPtrVector<char> EncodeFloat(SharedCudaPtr<float> data)
	{
		return this->Encode<float>(data);
	}

	SharedCudaPtr<float> DecodeFloat(SharedCudaPtrVector<char> data)
	{
		return this->Decode<float>(data);
	}

public:
	template<typename T> SharedCudaPtrVector<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtrVector<char> data);

private:
	template<typename T> SharedCudaPtr<T> FindUnique(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<char> CompressUnique(SharedCudaPtr<T> data, SharedCudaPtr<T> unique);
	template<typename T> SharedCudaPtr<T> DecompressUnique(SharedCudaPtr<char> data);

private:
	ExecutionPolicy _policy;

	friend class DictEncoding;
	friend class MostFrequentTest;
	FRIEND_TEST(MostFrequentTest, CompressDecompressMostFrequent_random_int);
};

} /* namespace ddj */

#endif /* UNIQUE_ENCODING_HPP_ */
