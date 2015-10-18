/*
 *  encoding_factory.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_ENCODING_FACTORY_HPP_
#define DDJ_ENCODING_FACTORY_HPP_

#include "encoding.hpp"
#include "encoding_type.hpp"
#include "data_type.hpp"
#include "compression/delta/delta_encoding.hpp"
#include "compression/none/none_encoding.hpp"
#include "compression/scale/scale_encoding.hpp"
#include "compression/rle/rle_encoding.hpp"
#include "compression/dict/dict_encoding.hpp"
#include "compression/unique/unique_encoding.hpp"
#include "compression/patch/patch_encoding.hpp"
#include "core/not_implemented_exception.hpp"
#include "core/operators.cuh"

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace ddj {

class EncodingFactory
{
public:
	static boost::shared_ptr<Encoding> Get(EncodingType encodingType)
    {
		OutsideOperator<int> op{500, 5000};
        switch(encodingType)
        {
			case EncodingType::delta:
                return boost::make_shared<DeltaEncoding>();
			case EncodingType::none:
				return boost::make_shared<NoneEncoding>();
			case EncodingType::scale:
				return boost::make_shared<ScaleEncoding>();
			case EncodingType::rle:
				return boost::make_shared<RleEncoding>();
			case EncodingType::dict:
				return boost::make_shared<DictEncoding>();
			case EncodingType::unique:
				return boost::make_shared<UniqueEncoding>();
			case EncodingType::patch:
				return boost::make_shared<PatchEncoding<OutsideOperator<int>>>(op);
			default:
				throw NotImplementedException("Encoding of this type not implemented");
        }
    }
};

} /* namespace ddj */
#endif /* DDJ_ENCODING_FACTORY_HPP_ */
