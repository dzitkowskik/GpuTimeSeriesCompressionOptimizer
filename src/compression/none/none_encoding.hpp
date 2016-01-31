/*
 *  none_encoding.cuh
 *
 *  Created on: 15/10/2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_NONE_ENCODING_HPP_
#define DDJ_NONE_ENCODING_HPP_

#include "core/cuda_ptr.hpp"
#include "compression/encoding.hpp"
#include "core/execution_policy.hpp"

#include "core/not_implemented_exception.hpp"
#include "compression/encoding_factory.hpp"
#include "data_type.hpp"
#include "compression/encoding_type.hpp"

#include <boost/make_shared.hpp>

namespace ddj
{

class NoneEncoding : public Encoding
{
public:
	NoneEncoding() : Encoding("Encoding.None") {}
	~NoneEncoding(){}
	NoneEncoding(const NoneEncoding&) = default;
	NoneEncoding(NoneEncoding&&) = default;

public:
	unsigned int GetNumberOfResults() { return 0; }

	size_t GetMetadataSize(SharedCudaPtr<char> data, DataType type)
	{
		return sizeof(size_t);
	}

	size_t GetCompressedSize(SharedCudaPtr<char> data, DataType type)
	{
		return data->size();
	}

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
	ExecutionPolicy _policy;
};

class NoneEncodingFactory : public EncodingFactory
{
public:
	NoneEncodingFactory(DataType dt)
		: EncodingFactory(dt, EncodingType::none)
	{}
	~NoneEncodingFactory(){}
	NoneEncodingFactory(const NoneEncodingFactory& other)
		: EncodingFactory(other.dataType, EncodingType::none)
	{}

	boost::shared_ptr<Encoding> Get()
	{
		return boost::make_shared<NoneEncoding>();
	}

	boost::shared_ptr<Encoding> Get(SharedCudaPtr<char> data)
	{
		return Get();
	}
};

} /* namespace ddj */
#endif /* DDJ_NONE_ENCODING_HPP_ */
