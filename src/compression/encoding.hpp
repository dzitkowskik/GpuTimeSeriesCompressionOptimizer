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
	SharedCudaPtrVector<char> Encode(SharedCudaPtr<char> data, DataType type)
	{
		switch(type)
		{
			case DataType::d_int:
				return EncodeInt(boost::reinterpret_pointer_cast<CudaPtr<int>>(data));
			case DataType::d_float:
				return EncodeFloat(boost::reinterpret_pointer_cast<CudaPtr<float>>(data));
			default:
				throw std::runtime_error("Encoding not implemented for this type");
		}
	}

	SharedCudaPtr<char> Decode(SharedCudaPtrVector<char> data, DataType type)
	{
		switch(type)
		{
			case DataType::d_int:
				return boost::reinterpret_pointer_cast<CudaPtr<char>>(DecodeInt(data));
			case DataType::d_float:
				return boost::reinterpret_pointer_cast<CudaPtr<char>>(DecodeFloat(data));
			default:
				throw std::runtime_error("Decoding not implemented for this type");
		}
	}

	virtual unsigned int GetNumberOfResults() = 0;

	// TODO: make abstract
	virtual size_t GetMetadataSize(SharedCudaPtr<char> data, DataType type) { return 0; }
	virtual size_t GetCompressedSize(SharedCudaPtr<char> data, DataType type) { return 0; }

protected:
	// INT
	virtual SharedCudaPtrVector<char> EncodeInt(SharedCudaPtr<int> data) = 0;
	virtual SharedCudaPtr<int> DecodeInt(SharedCudaPtrVector<char> data) = 0;

	// FLOAT
	virtual SharedCudaPtrVector<char> EncodeFloat(SharedCudaPtr<float> data) = 0;
	virtual SharedCudaPtr<float> DecodeFloat(SharedCudaPtrVector<char> data) = 0;
};

} /* namespace ddj */
#endif /* DDJ_ENCODING_HPP_ */
