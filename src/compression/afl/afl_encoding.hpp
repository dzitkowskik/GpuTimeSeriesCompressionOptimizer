/*
 *  afl_encoding.hpp
 *
 *  Created on: 25-10-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_AFL_ENCODING_HPP_
#define DDJ_AFL_ENCODING_HPP_

#include "compression/encoding.hpp"
#include "core/execution_policy.hpp"

namespace ddj
{

class AflEncoding : public Encoding
{
public:
	AflEncoding(){}
	~AflEncoding(){}
	AflEncoding(const AflEncoding&) = default;
	AflEncoding(AflEncoding&&) = default;

public:
	unsigned int GetNumberOfResults() { return 1; }

	size_t GetMetadataSize(SharedCudaPtr<char> data, DataType type)
	{
		return 2*sizeof(char);
	}

	size_t GetCompressedSize(SharedCudaPtr<char> data, DataType type);

protected:
	SharedCudaPtrVector<char> EncodeInt(SharedCudaPtr<int> data);
	SharedCudaPtr<int> DecodeInt(SharedCudaPtrVector<char> data);
	SharedCudaPtrVector<char> EncodeFloat(SharedCudaPtr<float> data);
	SharedCudaPtr<float> DecodeFloat(SharedCudaPtrVector<char> data);

public:
	template<typename T> SharedCudaPtrVector<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtrVector<char> data);

private:
	ExecutionPolicy _policy;
};

} /* namespace ddj */
#endif /* DDJ_AFL_ENCODING_HPP_ */
