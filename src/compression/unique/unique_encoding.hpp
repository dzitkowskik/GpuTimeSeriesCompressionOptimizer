/*
 * unique_encoding.hpp
 *
 *  Created on: 8 paź 2015
 *      Author: ghash
 */

#ifndef UNIQUE_ENCODING_HPP_
#define UNIQUE_ENCODING_HPP_

#include "core/cuda_ptr.hpp"
#include "core/execution_policy.hpp"
#include "compression/encoding.hpp"
#include <gtest/gtest.h>

#include "core/not_implemented_exception.hpp"
#include "compression/encoding_factory.hpp"
#include "data/data_type.hpp"
#include "compression/encoding_type.hpp"

#include <boost/make_shared.hpp>

namespace ddj {

class UniqueEncoding : public Encoding
{
public:
	UniqueEncoding() : Encoding("Encoding.Unique") {}
	~UniqueEncoding() {}
	UniqueEncoding(const UniqueEncoding&) = default;
	UniqueEncoding(UniqueEncoding&&) = default;

public:
	unsigned int GetNumberOfResults() { return 1; }

	size_t GetMetadataSize(SharedCudaPtr<char> data, DataType type);
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
	template<typename T> SharedCudaPtr<T> FindUnique(SharedCudaPtr<T> data);
	SharedCudaPtr<char> FindUnique(SharedCudaPtr<char> data, DataType type);

	template<typename T> SharedCudaPtr<char> CompressUnique(SharedCudaPtr<T> data, SharedCudaPtr<T> unique);
	template<typename T> SharedCudaPtr<T> DecompressUnique(SharedCudaPtr<char> data);

private:
	ExecutionPolicy _policy;

	friend class DictEncoding;
	friend class MostFrequentTest;
	FRIEND_TEST(MostFrequentTest, CompressDecompressMostFrequent_random_int);
};

class UniqueEncodingFactory : public EncodingFactory
{
public:
	UniqueEncodingFactory(DataType dt)
		: EncodingFactory(dt, EncodingType::unique)
	{}
	~UniqueEncodingFactory(){}
	UniqueEncodingFactory(const UniqueEncodingFactory& other)
		: EncodingFactory(other.dataType, EncodingType::unique)
	{}

	boost::shared_ptr<Encoding> Get()
	{
		return boost::make_shared<UniqueEncoding>();
	}

	boost::shared_ptr<Encoding> Get(SharedCudaPtr<char> data)
	{
		return Get();
	}
};

} /* namespace ddj */

#endif /* UNIQUE_ENCODING_HPP_ */
