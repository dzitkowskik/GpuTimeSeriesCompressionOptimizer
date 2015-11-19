/*
 * gfc_encoding.hpp
 *
 *  Created on: Nov 17, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef GFC_ENCODING_HPP_
#define GFC_ENCODING_HPP_

#include "compression/encoding.hpp"
#include "compression/encoding_factory.hpp"
#include "core/execution_policy.hpp"
#include "core/not_implemented_exception.hpp"

namespace ddj
{

class GfcEncoding : public Encoding
{
public:
	GfcEncoding() {}
	~GfcEncoding() {}
	GfcEncoding(const GfcEncoding&) = default;
	GfcEncoding(GfcEncoding&&) = default;

public:
	unsigned int GetNumberOfResults() { return 1; }

	size_t GetMetadataSize(SharedCudaPtr<char> data, DataType type)
	{ return 0; }

	size_t GetCompressedSize(SharedCudaPtr<char> data, DataType type)
	{ return 0; }

protected:
	SharedCudaPtrVector<char> EncodeInt(SharedCudaPtr<int> data);
	SharedCudaPtr<int> DecodeInt(SharedCudaPtrVector<char> data);
	SharedCudaPtrVector<char> EncodeFloat(SharedCudaPtr<float> data);
	SharedCudaPtr<float> DecodeFloat(SharedCudaPtrVector<char> data);
	SharedCudaPtrVector<char> EncodeDouble(SharedCudaPtr<double> data);
	SharedCudaPtr<double> DecodeDouble(SharedCudaPtrVector<char> data);

public:
	template<typename T> SharedCudaPtrVector<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtrVector<char> data);

private:
	ExecutionPolicy _policy;
};

} /* namespace ddj */

#endif /* GFC_ENCODING_HPP_ */