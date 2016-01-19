/*
 * lower_patch_encoding.hpp
 *
 *  Created on: Nov 14, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef LOWER_PATCH_ENCODING_HPP_
#define LOWER_PATCH_ENCODING_HPP_

#include "compression/patch/patch_encoding.hpp"

namespace ddj
{

class LowerPatchEncoding : public PatchEncoding
{
public:
	LowerPatchEncoding() : PatchEncoding(PatchType::lower) {}
	LowerPatchEncoding(double min, double max, double factor=0.1)
		: _min(min), _max(max), _factor(factor), PatchEncoding(PatchType::lower)
	{}
	~LowerPatchEncoding() {}
	LowerPatchEncoding(const LowerPatchEncoding& other)
		: _min(other._min), _max(other._max), _factor(other._factor),
		  PatchEncoding(PatchType::lower)
	{}
	LowerPatchEncoding(LowerPatchEncoding&& other)
		: _min(std::move(other._min)), _max(std::move(other._max)),
		  _factor(std::move(other._factor)), PatchEncoding(PatchType::lower)
	{}

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

	// TODO: Should be also private
public:
	template<typename T> SharedCudaPtrVector<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtrVector<char> data);

private:
	template<typename T> LowerOperator<T> GetOperator();

private:
	double _factor;
	double _min;
	double _max;
};

} /* namespace ddj */
#endif /* LOWER_PATCH_ENCODING_HPP_ */
