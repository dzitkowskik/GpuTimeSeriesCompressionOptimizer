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

#include <boost/make_shared.hpp>

namespace ddj
{

class ConstEncoding : public Encoding
{
public:
	ConstEncoding(){}
	virtual ~ConstEncoding(){}
	ConstEncoding(const ConstEncoding&) = default;

public:
	unsigned int GetNumberOfResults() { return 1; }

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

public:
	template<typename T> SharedCudaPtrVector<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtrVector<char> data);

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
