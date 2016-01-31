/*
 *  const_encoding.hpp
 *
 *  Created on: 30-10-2015
 *      Author: Karol Dzitkowski
 */

#ifndef CONST_ENCODING_HPP_
#define CONST_ENCODING_HPP_

#include "compression/encoding.hpp"
#include "compression/encoding_factory.hpp"
#include "core/execution_policy.hpp"
#include "core/not_implemented_exception.hpp"
#include "util/stencil/stencil.hpp"

#include <boost/make_shared.hpp>

namespace ddj
{

class ConstEncoding : public Encoding
{
public:
	ConstEncoding() : Encoding("Encoding.Const") {}
	virtual ~ConstEncoding(){}
	ConstEncoding(const ConstEncoding&) = default;

public:
	unsigned int GetNumberOfResults() { return 1; }

	size_t GetMetadataSize(SharedCudaPtr<char> data, DataType type)
	{
		if(data->size() <= 0) return 0;
		int elemCnt = data->size() / GetDataTypeSize(type);
		return (elemCnt + 7) / 8 + GetDataTypeSize(type) + 1;
	}

	size_t GetCompressedSize(SharedCudaPtr<char> data, DataType type);

protected:
	SharedCudaPtrVector<char> EncodeInt(SharedCudaPtr<int> data)
	{ return this->Encode<int>(data); }
	SharedCudaPtr<int> DecodeInt(SharedCudaPtrVector<char> data)
	{ return this->Decode<int>(data); }
	SharedCudaPtrVector<char> EncodeTime(SharedCudaPtr<time_t> data)
	{ return this->Encode<time_t>(data); }
	SharedCudaPtr<time_t> DecodeTime(SharedCudaPtrVector<char> data)
	{ return this->Decode<time_t>(data); }
	SharedCudaPtrVector<char> EncodeFloat(SharedCudaPtr<float> data)
	{ return this->Encode<float>(data); }
	SharedCudaPtr<float> DecodeFloat(SharedCudaPtrVector<char> data)
	{ return this->Decode<float>(data); }
	SharedCudaPtrVector<char> EncodeDouble(SharedCudaPtr<double> data)
	{ return this->Encode<double>(data); }
	SharedCudaPtr<double> DecodeDouble(SharedCudaPtrVector<char> data)
	{ return this->Decode<double>(data); }
	SharedCudaPtrVector<char> EncodeShort(SharedCudaPtr<short> data)
	{ return this->Encode<short>(data); }
	SharedCudaPtr<short> DecodeShort(SharedCudaPtrVector<char> data)
	{ return this->Decode<short>(data); }
	SharedCudaPtrVector<char> EncodeChar(SharedCudaPtr<char> data)
	{ return this->Encode<char>(data); }
	SharedCudaPtr<char> DecodeChar(SharedCudaPtrVector<char> data)
	{ return this->Decode<char>(data); }

public:
	template<typename T> SharedCudaPtrVector<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtrVector<char> data);

private:
	template<typename T> size_t GetCompressedSize(SharedCudaPtr<T> data);
	template<typename T> Stencil GetConstStencil(SharedCudaPtr<T> data, T constValue);

private:
	ExecutionPolicy _policy;
};

class ConstEncodingFactory : public EncodingFactory
{
public:
	ConstEncodingFactory(DataType dt)
		: EncodingFactory(dt, EncodingType::constData)
	{}
	~ConstEncodingFactory(){}
	ConstEncodingFactory(const ConstEncodingFactory& other)
		: EncodingFactory(other.dataType, EncodingType::constData)
	{}

	boost::shared_ptr<Encoding> Get()
	{
		return boost::make_shared<ConstEncoding>();
	}

	boost::shared_ptr<Encoding> Get(SharedCudaPtr<char> data)
	{
		return Get();
	}
};

} /* namespace ddj */

#endif /* CONST_ENCODING_HPP_ */
