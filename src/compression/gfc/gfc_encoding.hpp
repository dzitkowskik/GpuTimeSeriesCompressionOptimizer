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
	unsigned int GetNumberOfResults() { return 2; }

	size_t GetMetadataSize(SharedCudaPtr<char> data, DataType type)
	{ return 3*sizeof(int); }

	size_t GetCompressedSize(SharedCudaPtr<char> data, DataType type)
	{ return 0; }

protected:
	SharedCudaPtrVector<char> EncodeInt(SharedCudaPtr<int> data);
	SharedCudaPtr<int> DecodeInt(SharedCudaPtrVector<char> data);
	SharedCudaPtrVector<char> EncodeTime(SharedCudaPtr<time_t> data);
	SharedCudaPtr<time_t> DecodeTime(SharedCudaPtrVector<char> data);
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

class GfcEncodingFactory : public EncodingFactory
{
public:
	GfcEncodingFactory(DataType dt)
		: EncodingFactory(dt, EncodingType::gfc)
	{}
	~GfcEncodingFactory(){}
	GfcEncodingFactory(const GfcEncodingFactory& other)
		: EncodingFactory(other.dataType, EncodingType::gfc)
	{}

	boost::shared_ptr<Encoding> Get()
	{
		return boost::make_shared<GfcEncoding>();
	}

	boost::shared_ptr<Encoding> Get(SharedCudaPtr<char> data)
	{
		return Get();
	}
};

} /* namespace ddj */

#endif /* GFC_ENCODING_HPP_ */
