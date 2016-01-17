/*
 * scale_encoding.cuh
 *
 *  Created on: 24-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_SCALE_ENCODING_HPP_
#define DDJ_SCALE_ENCODING_HPP_

#include "compression/encoding.hpp"
#include "compression/encoding_factory.hpp"

#include "core/execution_policy.hpp"
#include "core/not_implemented_exception.hpp"

#include <boost/make_shared.hpp>

namespace ddj {

class ScaleEncoding : public Encoding
{
public:
	ScaleEncoding(){}
	~ScaleEncoding(){}
	ScaleEncoding(const ScaleEncoding&) = default;

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
	{ return this->Encode<int>(CastSharedCudaPtr<float, int>(data)); }
	SharedCudaPtr<float> DecodeFloat(SharedCudaPtrVector<char> data)
	{ return CastSharedCudaPtr<int, float>(this->Decode<int>(data)); }
	SharedCudaPtrVector<char> EncodeDouble(SharedCudaPtr<double> data)
	{ return this->Encode<time_t>(CastSharedCudaPtr<double, time_t>(data)); }
	SharedCudaPtr<double> DecodeDouble(SharedCudaPtrVector<char> data)
	{ return CastSharedCudaPtr<time_t, double>(this->Decode<time_t>(data)); }

public:
	template<typename T> SharedCudaPtrVector<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtrVector<char> data);

private:
	ExecutionPolicy _policy;
};

class ScaleEncodingFactory : public EncodingFactory
{
public:
	ScaleEncodingFactory(DataType dt)
		: EncodingFactory(dt, EncodingType::scale)
	{}
	~ScaleEncodingFactory(){}
	ScaleEncodingFactory(const ScaleEncodingFactory& other)
		: EncodingFactory(other.dataType, EncodingType::scale)
	{}

	boost::shared_ptr<Encoding> Get()
	{
		return boost::make_shared<ScaleEncoding>();
	}

	boost::shared_ptr<Encoding> Get(SharedCudaPtr<char> data)
	{
		return Get();
	}
};

} /* namespace ddj */
#endif /* DDJ_SCALE_ENCODING_HPP_ */
