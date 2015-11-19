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

#include "core/not_implemented_exception.hpp"
#include "compression/encoding_factory.hpp"
#include "compression/data_type.hpp"
#include "compression/encoding_type.hpp"

#include <boost/make_shared.hpp>


namespace ddj {

class DeltaEncoding : public Encoding
{
public:
	DeltaEncoding(){}
	~DeltaEncoding(){}
	DeltaEncoding(const DeltaEncoding&) = default;
	DeltaEncoding(DeltaEncoding&&) = default;

public:
	unsigned int GetNumberOfResults() { return 1; }

	size_t GetMetadataSize(SharedCudaPtr<char> data, DataType type)
	{
		if(data->size() <= 0) return 0;
		return GetDataTypeSize(type);
	}

	size_t GetCompressedSize(SharedCudaPtr<char> data, DataType type)
	{
		if(data->size() <= 0) return 0;
		return data->size() - GetDataTypeSize(type);
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

class DeltaEncodingFactory : public EncodingFactory
{
public:
	DeltaEncodingFactory(DataType dt)
		: EncodingFactory(dt, EncodingType::delta)
	{}
	~DeltaEncodingFactory(){}
	DeltaEncodingFactory(const DeltaEncodingFactory& other)
		: EncodingFactory(other.dataType, EncodingType::delta)
	{}

	boost::shared_ptr<Encoding> Get()
	{
		return boost::make_shared<DeltaEncoding>();
	}

	boost::shared_ptr<Encoding> Get(SharedCudaPtr<char> data)
	{
		return Get();
	}
};

} /* namespace ddj */
#endif /* DDJ_DELTA_ENCODING_HPP_ */
