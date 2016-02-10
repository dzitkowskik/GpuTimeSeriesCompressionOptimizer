/*
 *  float_encoding.hpp
 *
 *  Created on: 30 paź 2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_FLOAT_ENCODING_HPP_
#define DDJ_FLOAT_ENCODING_HPP_

#include "compression/encoding.hpp"
#include "compression/encoding_factory.hpp"

#include "core/execution_policy.hpp"
#include "core/not_implemented_exception.hpp"

#include <boost/make_shared.hpp>

namespace ddj
{

class FloatEncoding : public Encoding
{
public:
	FloatEncoding() : Encoding("Encoding.Float") {}
	virtual ~FloatEncoding(){}
	FloatEncoding(const FloatEncoding&) = default;

public:
	unsigned int GetNumberOfResults() { return 1; }
	std::vector<DataType> GetReturnTypes(DataType type) { return std::vector<DataType>{DataType::d_int}; }

	size_t GetMetadataSize(SharedCudaPtr<char> data, DataType type)
	{
		if(data->size() <= 0) return 0;
		return sizeof(int);
	}

	size_t GetCompressedSize(SharedCudaPtr<char> data, DataType type)
	{
		// TODO: For 8 bytes types result size is data->size() / 2;
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

class FloatEncodingFactory : public EncodingFactory
{
public:
	FloatEncodingFactory(DataType dt)
		: EncodingFactory(dt, EncodingType::floatToInt)
	{}
	~FloatEncodingFactory(){}
	FloatEncodingFactory(const FloatEncodingFactory& other)
		: EncodingFactory(other.dataType, EncodingType::floatToInt)
	{}

	boost::shared_ptr<Encoding> Get()
	{
		return boost::make_shared<FloatEncoding>();
	}

	boost::shared_ptr<Encoding> Get(SharedCudaPtr<char> data)
	{
		return Get();
	}
};

} /* namespace ddj */

#endif /* DDJ_FLOAT_ENCODING_HPP_ */
