/*
 *  encoding.hpp
 *
 *  Created on: 10-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_ENCODING_HPP_
#define DDJ_ENCODING_HPP_

#include "core/cuda_ptr.hpp"
#include "compression/data_type.hpp"
#include <boost/pointer_cast.hpp>

namespace ddj {

class Encoding
{
public:
	Encoding(){}
	virtual ~Encoding(){}

public:
	SharedCudaPtr<char> Encode(SharedCudaPtr<char> data, DataType type)
	{
		switch(type)
		{
			case DataType::d_int:
				return EncodeInt(MoveSharedCudaPtr<char, int>(data));
			case DataType::d_float:
				return EncodeFloat(MoveSharedCudaPtr<char, float>(data));
			default:
				throw std::runtime_error("Encoding not implemented for this type");
		}
	}

	SharedCudaPtr<char> Decode(SharedCudaPtr<char> data, DataType type)
	{
		switch(type)
		{
			case DataType::d_int:
				return MoveSharedCudaPtr<int, char>(DecodeInt(data));
			case DataType::d_float:
				return MoveSharedCudaPtr<float, char>(DecodeFloat(data));
			default:
				throw std::runtime_error("Decoding not implemented for this type");
		}
	}

protected:
	virtual SharedCudaPtr<char> EncodeInt(SharedCudaPtr<int> data) = 0;
	virtual SharedCudaPtr<int> DecodeInt(SharedCudaPtr<char> data) = 0;
	virtual SharedCudaPtr<char> EncodeFloat(SharedCudaPtr<float> data) = 0;
	virtual SharedCudaPtr<float> DecodeFloat(SharedCudaPtr<char> data) = 0;
};

} /* namespace ddj */
#endif /* DDJ_ENCODING_HPP_ */
