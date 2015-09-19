/*
 *  delta_encoding.hpp
 *
 *  Created on: 18-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_DELTA_ENCODING_HPP_
#define DDJ_DELTA_ENCODING_HPP_

#include "compression/encoding.hpp"
#include "core/cuda_ptr.hpp"
#include "core/execution_policy.hpp"

namespace ddj {

class DeltaEncoding : public Encoding
{
public:
	SharedCudaPtrVector<char> EncodeInt(SharedCudaPtr<int> data) override
	{ return this.Encode<int>(data); }
	SharedCudaPtr<int> DecodeInt(SharedCudaPtrVector<char> data) override
	{ return this.Decode<int>(data); }
	SharedCudaPtrVector<char> EncodeFloat(SharedCudaPtr<float> data) override;
	SharedCudaPtr<float> DecodeFloat(SharedCudaPtrVector<char> data) override;

private:
	template<typename T> SharedCudaPtrVector<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtrVector<char> data);

private:
	ExecutionPolicy _policy;
};

} /* namespace ddj */
#endif /* DDJ_DELTA_ENCODING_HPP_ */
