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
#include "compression/data_type.hpp"
#include "compression/encoding_type.hpp"

#include <boost/make_shared.hpp>

namespace ddj
{

class NoneEncoding : public Encoding
{
public:
	NoneEncoding(){}
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

	SharedCudaPtrVector<char> EncodeDouble(SharedCudaPtr<double> data)
	{ return SharedCudaPtrVector<char>(); }
	SharedCudaPtr<double> DecodeDouble(SharedCudaPtrVector<char> data)
	{ return SharedCudaPtr<double>(); }

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
