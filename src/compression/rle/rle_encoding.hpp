/*
 *  rle_encoding.hpp
 *
 *  Created on: 9/10/2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_RLE_ENCODING_HPP_
#define DDJ_RLE_ENCODING_HPP_

#include "core/cuda_ptr.hpp"
#include "core/execution_policy.hpp"
#include "compression/encoding.hpp"

#include "core/not_implemented_exception.hpp"
#include "compression/encoding_factory.hpp"
#include "data_type.hpp"
#include "compression/encoding_type.hpp"

#include <boost/make_shared.hpp>

namespace ddj {

// TODO: Try version with returning 2 results (one for lengths and other for data)
class RleEncoding : public Encoding
{
public:
	RleEncoding() : Encoding("Encoding.Rle") {}
	~RleEncoding(){}
	RleEncoding(const RleEncoding&) = default;
	RleEncoding(RleEncoding&&) = default;

public:
	unsigned int GetNumberOfResults() { return 2; }

	size_t GetMetadataSize(SharedCudaPtr<char> data, DataType type)
	{
		if(data->size() <= 0) return 0;
		return sizeof(int);
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

	template<typename T>
	size_t GetCompressedSize(SharedCudaPtr<T> data);

public:
	template<typename T> SharedCudaPtrVector<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtrVector<char> data);

private:
	ExecutionPolicy _policy;
};

class RleEncodingFactory : public EncodingFactory
{
public:
	RleEncodingFactory(DataType dt)
		: EncodingFactory(dt, EncodingType::rle)
	{}
	~RleEncodingFactory(){}
	RleEncodingFactory(const RleEncodingFactory& other)
		: EncodingFactory(other.dataType, EncodingType::rle)
	{}

	boost::shared_ptr<Encoding> Get()
	{
		return boost::make_shared<RleEncoding>();
	}

	boost::shared_ptr<Encoding> Get(SharedCudaPtr<char> data)
	{
		return Get();
	}
};

} /* namespace ddj */

#endif /* DDJ_RLE_ENCODING_HPP_ */
