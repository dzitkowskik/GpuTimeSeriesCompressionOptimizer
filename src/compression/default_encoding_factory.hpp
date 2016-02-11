/*
 *  default_encoding_factory.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_DEFAULT_ENCODING_FACTORY_HPP_
#define DDJ_DEFAULT_ENCODING_FACTORY_HPP_

#include "compression/encoding.hpp"
#include "compression/encoding_type.hpp"
#include "core/not_implemented_exception.hpp"

#include "compression/delta/delta_encoding.hpp"
#include "compression/dict/dict_encoding.hpp"
#include "compression/none/none_encoding.hpp"
#include "compression/patch/patch_encoding_factory.hpp"
#include "compression/rle/rle_encoding.hpp"
#include "compression/scale/scale_encoding.hpp"
#include "compression/unique/unique_encoding.hpp"
#include "compression/afl/afl_encoding.hpp"
#include "compression/const/const_encoding.hpp"
#include "compression/float/float_encoding.hpp"
#include "compression/gfc/gfc_encoding.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace ddj {

class DefaultEncodingFactory
{

public:
	static boost::shared_ptr<EncodingFactory> Get(EncodingType encodingType, DataType dataType)
	{
		switch(encodingType)
		{
			case EncodingType::delta:
				return boost::make_shared<DeltaEncodingFactory>(dataType);
			case EncodingType::dict:
				return boost::make_shared<DictEncodingFactory>(dataType);
			case EncodingType::none:
				return boost::make_shared<NoneEncodingFactory>(dataType);
			case EncodingType::patch:
				switch(dataType)
				{
					case DataType::d_int:
						return boost::make_shared<PatchEncodingFactory<int>>(dataType, PatchType::lower);
					case DataType::d_time:
						return boost::make_shared<PatchEncodingFactory<time_t>>(dataType, PatchType::lower);
					case DataType::d_float:
						return boost::make_shared<PatchEncodingFactory<float>>(dataType, PatchType::lower);
					case DataType::d_double:
						return boost::make_shared<PatchEncodingFactory<double>>(dataType, PatchType::lower);
					case DataType::d_short:
						return boost::make_shared<PatchEncodingFactory<short>>(dataType, PatchType::lower);
					case DataType::d_char:
						return boost::make_shared<PatchEncodingFactory<char>>(dataType, PatchType::lower);
					default:
						throw NotImplementedException("Encoding of this type not implemented");
				}
				break;
			case EncodingType::scale:
				return boost::make_shared<ScaleEncodingFactory>(dataType);
			case EncodingType::rle:
				return boost::make_shared<RleEncodingFactory>(dataType);
			case EncodingType::unique:
				return boost::make_shared<UniqueEncodingFactory>(dataType);
			case EncodingType::afl:
				return boost::make_shared<AflEncodingFactory>(dataType);
			case EncodingType::gfc:
				return boost::make_shared<GfcEncodingFactory>(dataType);
			case EncodingType::constData:
				return boost::make_shared<ConstEncodingFactory>(dataType);
			case EncodingType::floatToInt:
				return boost::make_shared<FloatEncodingFactory>(dataType);
			default:
				throw NotImplementedException("Encoding of this type not implemented");
		}
	}
};

} /* namespace ddj */
#endif /* DDJ_DEFAULT_ENCODING_FACTORY_HPP_ */
