/*
 * outside_patch_encoding.hpp
 *
 *  Created on: Nov 14, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef OUTSIDE_PATCH_ENCODING_HPP_
#define OUTSIDE_PATCH_ENCODING_HPP_

#include "compression/patch/patch_encoding.hpp"

namespace ddj
{

class OutsidePatchEncoding : public PatchEncoding
{
public:
	OutsidePatchEncoding() : PatchEncoding(PatchType::outside) {}
	OutsidePatchEncoding(double min, double max, double factor=0.1)
		: _min(min), _max(max), _factor(factor), PatchEncoding(PatchType::outside)
	{}
	~OutsidePatchEncoding() {}
	OutsidePatchEncoding(const OutsidePatchEncoding& other)
		: _min(other._min), _max(other._max), _factor(other._factor),
		  PatchEncoding(PatchType::outside)
	{}
	OutsidePatchEncoding(OutsidePatchEncoding&& other)
		: _min(std::move(other._min)), _max(std::move(other._max)),
		  _factor(std::move(other._factor)), PatchEncoding(PatchType::outside)
	{}

protected:
	SharedCudaPtrVector<char> EncodeInt(SharedCudaPtr<int> data)
	{
		return this->Encode<int>(data);
	}
	SharedCudaPtrVector<char> EncodeFloat(SharedCudaPtr<float> data)
	{
		return this->Encode<float>(data);
	}
	SharedCudaPtr<int> DecodeInt(SharedCudaPtrVector<char> data)
	{
		return this->Decode<int>(data);
	}
	SharedCudaPtr<float> DecodeFloat(SharedCudaPtrVector<char> data)
	{
		return this->Decode<float>(data);
	}

	SharedCudaPtrVector<char> EncodeDouble(SharedCudaPtr<double> data)
	{ return SharedCudaPtrVector<char>(); }
	SharedCudaPtr<double> DecodeDouble(SharedCudaPtrVector<char> data)
	{ return SharedCudaPtr<double>(); }

	// TODO: Should be also private
public:
	template<typename T> SharedCudaPtrVector<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtrVector<char> data);

private:
	template<typename T> OutsideOperator<T> GetOperator();

private:
	double _factor;
	double _min;
	double _max;
};

} /* namespace ddj */

#endif /* OUTSIDE_PATCH_ENCODING_HPP_ */
