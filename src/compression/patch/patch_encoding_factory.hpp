/*
 * patch_encoding_factory.hpp
 *
 *  Created on: Nov 14, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef PATCH_ENCODING_FACTORY_HPP_
#define PATCH_ENCODING_FACTORY_HPP_

#include "compression/encoding_factory.hpp"
#include "compression/patch/outside_patch_encoding.hpp"
#include "compression/patch/lower_patch_encoding.hpp"

#include <boost/make_shared.hpp>


namespace ddj {

template<typename T>
class PatchEncodingFactory : public EncodingFactory
{
public:
	PatchType patchType;
	T factor;

	PatchEncodingFactory(DataType dt, PatchType pt)
		: EncodingFactory(dt, EncodingType::patch), patchType(pt)
	{ factor = 0.1; }
	~PatchEncodingFactory(){}
	PatchEncodingFactory(const PatchEncodingFactory& other)
		: EncodingFactory(other.dataType, EncodingType::patch), patchType(other.patchType)
	{}

	boost::shared_ptr<Encoding> Get()
	{
		return boost::make_shared<OutsidePatchEncoding>(0, 0, 0);
	}

	boost::shared_ptr<Encoding> Get(SharedCudaPtr<char> data)
	{
		if(data->size() <= 0) return Get();

		auto minMax = CudaArrayStatistics().MinMax(CastSharedCudaPtr<char, T>(data));
		T min = std::get<0>(minMax);
		T max = std::get<1>(minMax);

		switch(patchType)
		{
			case PatchType::outside:
				return boost::make_shared<OutsidePatchEncoding>(min, max, factor);
			case PatchType::lower:
				return boost::make_shared<LowerPatchEncoding>(min, max, factor);
			default:
				throw NotImplementedException("Patch Encoding of this type not implemented");
		}
	}
};

} /* namespace ddj */
#endif /* PATCH_ENCODING_FACTORY_HPP_ */
