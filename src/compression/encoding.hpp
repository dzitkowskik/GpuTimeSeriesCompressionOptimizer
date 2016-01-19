/*
 *  encoding.hpp
 *
 *  Created on: 10-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_ENCODING_HPP_
#define DDJ_ENCODING_HPP_

#include "core/cuda_ptr.hpp"
#include "data_type.hpp"
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
			case DataType::d_time:
				return EncodeTime(boost::reinterpret_pointer_cast<CudaPtr<time_t>>(data));
			case DataType::d_float:
				return EncodeFloat(boost::reinterpret_pointer_cast<CudaPtr<float>>(data));
			case DataType::d_double:
				return EncodeDouble(boost::reinterpret_pointer_cast<CudaPtr<double>>(data));
			case DataType::d_short:
				return EncodeShort(boost::reinterpret_pointer_cast<CudaPtr<short>>(data));
			case DataType::d_char:
				return EncodeChar(data);
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
			case DataType::d_time:
				return boost::reinterpret_pointer_cast<CudaPtr<char>>(DecodeTime(data));
			case DataType::d_float:
				return boost::reinterpret_pointer_cast<CudaPtr<char>>(DecodeFloat(data));
			case DataType::d_double:
				return boost::reinterpret_pointer_cast<CudaPtr<char>>(DecodeDouble(data));
			case DataType::d_short:
				return boost::reinterpret_pointer_cast<CudaPtr<char>>(DecodeShort(data));
			case DataType::d_char:
				return DecodeChar(data);
			default:
				throw std::runtime_error("Decoding not implemented for this type");
		}
	}

	virtual unsigned int GetNumberOfResults() = 0;
	virtual DataType GetReturnType(DataType type) { return type; }

	// TODO: make abstract
	virtual size_t GetMetadataSize(SharedCudaPtr<char> data, DataType type) = 0;
	virtual size_t GetCompressedSize(SharedCudaPtr<char> data, DataType type) = 0;

public:
	static inline double GetCompressionRatio(size_t inputSize, size_t outputSize)
	{
		if(outputSize <= 0 || outputSize > inputSize) return 1.0;
		double is = inputSize;
		double os = outputSize;
		return is/os;
	}

protected:
	// INT
	virtual SharedCudaPtrVector<char> EncodeInt(SharedCudaPtr<int> data) = 0;
	virtual SharedCudaPtr<int> DecodeInt(SharedCudaPtrVector<char> data) = 0;

	// TIME_T
	virtual SharedCudaPtrVector<char> EncodeTime(SharedCudaPtr<time_t> data) = 0;
	virtual SharedCudaPtr<time_t> DecodeTime(SharedCudaPtrVector<char> data) = 0;

	// FLOAT
	virtual SharedCudaPtrVector<char> EncodeFloat(SharedCudaPtr<float> data) = 0;
	virtual SharedCudaPtr<float> DecodeFloat(SharedCudaPtrVector<char> data) = 0;

	// DOUBLE
	virtual SharedCudaPtrVector<char> EncodeDouble(SharedCudaPtr<double> data) = 0;
	virtual SharedCudaPtr<double> DecodeDouble(SharedCudaPtrVector<char> data) = 0;

	// SHORT
	virtual SharedCudaPtrVector<char> EncodeShort(SharedCudaPtr<short> data) = 0;
	virtual SharedCudaPtr<short> DecodeShort(SharedCudaPtrVector<char> data) = 0;

	// CHAR
	virtual SharedCudaPtrVector<char> EncodeChar(SharedCudaPtr<char> data) = 0;
	virtual SharedCudaPtr<char> DecodeChar(SharedCudaPtrVector<char> data) = 0;
};

} /* namespace ddj */
#endif /* DDJ_ENCODING_HPP_ */
