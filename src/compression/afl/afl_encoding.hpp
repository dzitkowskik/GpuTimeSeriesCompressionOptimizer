/*
 *  afl_encoding.hpp
 *
 *  Created on: 25-10-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_AFL_ENCODING_HPP_
#define DDJ_AFL_ENCODING_HPP_

#include "compression/encoding.hpp"
#include "compression/encoding_factory.hpp"
#include "core/execution_policy.hpp"

namespace ddj
{

class AflEncoding : public Encoding
{
public:
	AflEncoding(){}
	~AflEncoding(){}
	AflEncoding(const AflEncoding&) = default;
	AflEncoding(AflEncoding&&) = default;

public:
	unsigned int GetNumberOfResults() { return 1; }

	size_t GetMetadataSize(SharedCudaPtr<char> data, DataType type);
	size_t GetCompressedSize(SharedCudaPtr<char> data, DataType type);

protected:
	SharedCudaPtrVector<char> EncodeInt(SharedCudaPtr<int> data);
	SharedCudaPtr<int> DecodeInt(SharedCudaPtrVector<char> data);
	SharedCudaPtrVector<char> EncodeTime(SharedCudaPtr<time_t> data);
	SharedCudaPtr<time_t> DecodeTime(SharedCudaPtrVector<char> data);
	SharedCudaPtrVector<char> EncodeFloat(SharedCudaPtr<float> data);
	SharedCudaPtr<float> DecodeFloat(SharedCudaPtrVector<char> data);
	SharedCudaPtrVector<char> EncodeDouble(SharedCudaPtr<double> data);
	SharedCudaPtr<double> DecodeDouble(SharedCudaPtrVector<char> data);

	template<typename T>
	size_t GetCompressedSizeIntegral(SharedCudaPtr<T> data);

	template<typename T>
	size_t GetCompressedSizeFloatingPoint(SharedCudaPtr<T> data);

public:
	template<typename T> SharedCudaPtrVector<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtrVector<char> data);

private:
	ExecutionPolicy _policy;
};

class AflEncodingFactory : public EncodingFactory
{
public:
	AflEncodingFactory(DataType dt)
		: EncodingFactory(dt, EncodingType::afl)
	{}
	~AflEncodingFactory(){}
	AflEncodingFactory(const AflEncodingFactory& other)
		: EncodingFactory(other.dataType, EncodingType::afl)
	{}

	boost::shared_ptr<Encoding> Get()
	{
		return boost::make_shared<AflEncoding>();
	}

	boost::shared_ptr<Encoding> Get(SharedCudaPtr<char> data)
	{
		return Get();
	}
};

} /* namespace ddj */
#endif /* DDJ_AFL_ENCODING_HPP_ */
