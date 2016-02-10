/*
 *  dict_encoding.hpp
 *
 *  Created on: 18-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_DICT_ENCODING_HPP_
#define DDJ_DICT_ENCODING_HPP_

#include "core/cuda_ptr.hpp"
#include "compression/encoding.hpp"
#include "core/execution_policy.hpp"
#include "util/splitter/splitter.hpp"
#include <gtest/gtest.h>

#include "core/not_implemented_exception.hpp"
#include "compression/encoding_factory.hpp"
#include "data_type.hpp"
#include "compression/encoding_type.hpp"

#include <boost/make_shared.hpp>

namespace ddj {

class DictEncoding : public Encoding
{
public:
	DictEncoding()
		: _freqCnt(4), Encoding("Encoding.Dict") {}
	~DictEncoding() {}
	DictEncoding(const DictEncoding& other)
		: _freqCnt(other._freqCnt), Encoding("DictEncoding")
	{};
	DictEncoding(DictEncoding&& other)
		: _freqCnt(std::move(other._freqCnt)), Encoding("DictEncoding")
	{};

public:
	void SetFreqCnt(int freqCnt) { _freqCnt = freqCnt; }
	unsigned int GetNumberOfResults() { return 2; }
	std::vector<DataType> GetReturnTypes(DataType type) { return std::vector<DataType>{type, type}; }

	size_t GetMetadataSize(SharedCudaPtr<char> data, DataType type)
	{
		int elemCnt = data->size() / GetDataTypeSize(type);
		return (elemCnt + 7) / 8 + 1;
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

private:
	template<typename T>
    SharedCudaPtr<int> GetMostFrequentStencil(SharedCudaPtr<T> data, SharedCudaPtr<T> mostFrequent);

	template<typename T>
	size_t GetCompressedSize(SharedCudaPtr<T> data);

// TODO: Should be also private
public:
	template<typename T> SharedCudaPtrVector<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtrVector<char> data);

private:
	int _freqCnt;

private:
	Splitter _splitter;
	ExecutionPolicy _policy;

	friend class MostFrequentTest;
	FRIEND_TEST(MostFrequentTest, CompressDecompressMostFrequent_random_int);
};

class DictEncodingFactory : public EncodingFactory
{
public:
	DictEncodingFactory(DataType dt)
		: EncodingFactory(dt, EncodingType::dict)
	{}
	~DictEncodingFactory(){}
	DictEncodingFactory(const DictEncodingFactory& other)
		: EncodingFactory(other.dataType, EncodingType::dict)
	{}

	boost::shared_ptr<Encoding> Get()
	{
		return boost::make_shared<DictEncoding>();
	}

	boost::shared_ptr<Encoding> Get(SharedCudaPtr<char> data)
	{
		return Get();
	}
};

} /* namespace ddj */
#endif /* DDJ_DICT_ENCODING_HPP_ */
