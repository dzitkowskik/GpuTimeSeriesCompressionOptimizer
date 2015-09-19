/*
 *  encoding.hpp
 *
 *  Created on: 10-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_ENCODING_HPP_
#define DDJ_ENCODING_HPP_

namespace ddj {

#include "core/cuda_ptr.hpp"
#include "data_type.hpp"
#include <boost/pointer_cast.hpp>

class Encoding
{
public:
	Encoding(DataType type) : _type(type) {}
	virtual ~Encoding();
	Encoding(const Encoding& other) : _type(other._type) {}
	Encoding(Encoding&& other) noexcept : _type(std::move(other._type)) {}

public:
	EncodingResult Encode(SharedCudaPtr<char> data)
	{
		switch(this._type)
		{
			case DataType.int:
				auto int_data = boost::reinterpret_pointer_cast<CudaPtr<int>>(data);
				return EncodeInt();
			case DataType.float:
				auto float_data = boost::reinterpret_pointer_cast<CudaPtr<float>>(data);
				return EncodeFloat(float_data);
			case default:
				throw std::runtime_error("Not implemented type for this encoding");
		}
	}

	SharedCudaPtr<char> Decode(EncodingResult data)
	{
		switch(this._type)
		{
			case DataType.int:
				return DecodeInt(data);
			case DataType.float:
				return DecodeFloat(data);
			case default:
				throw std::runtime_error("Not implemented type for this encoding");
		}
	}

protected:
	virtual SharedCudaPtrVector<char> EncodeInt(SharedCudaPtr<int> data) = 0;
	virtual SharedCudaPtr<int> DecodeInt(SharedCudaPtrVector<char> data) = 0;

	virtual SharedCudaPtrVector<char> EncodeFloat(SharedCudaPtr<float> data) = 0;
	virtual SharedCudaPtr<float> DecodeFloat(SharedCudaPtrVector<char> data) = 0;

private:
	DataType _type;
};

} /* namespace ddj */
#endif /* DDJ_ENCODING_HPP_ */
