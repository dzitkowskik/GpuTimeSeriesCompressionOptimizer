/*
 * scale_encoding.cuh
 *
 *  Created on: 24-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_SCALE_ENCODING_HPP_
#define DDJ_SCALE_ENCODING_HPP_

#include "core/cuda_ptr.hpp"
#include "compression/encoding.hpp"
#include "core/execution_policy.hpp"

namespace ddj {

class ScaleEncoding : public Encoding
{
public:
	ScaleEncoding(){}
	~ScaleEncoding(){}
	ScaleEncoding(const ScaleEncoding&) = default;

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
#endif /* DDJ_SCALE_ENCODING_HPP_ */
