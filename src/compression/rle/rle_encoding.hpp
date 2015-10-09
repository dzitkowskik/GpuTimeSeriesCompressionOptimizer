/*
 *  rle_encoding.hpp
 *
 *  Created on: 9/10/2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_RLE_ENCODING_HPP_
#define DDJ_RLE_ENCODING_HPP_

#include "core/cuda_ptr.hpp"
#include "core/execution_policy.hpp"
#include "compression/encoding.hpp"

namespace ddj {

// TODO: Try version with returning 2 results (one for lengths and other for data)
class RleEncoding : public Encoding
{
public:
	RleEncoding(){}
	~RleEncoding(){}
	RleEncoding(const RleEncoding&) = default;
	RleEncoding(RleEncoding&&) = default;

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
	ExecutionPolicy _policy;
};

} /* namespace ddj */

#endif /* DDJ_RLE_ENCODING_HPP_ */
