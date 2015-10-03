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
#include "core/not_implemented_exception.hpp"

namespace ddj {

class EncodingFactory
{
public:
	static Encoding Get(EncodingType encodingType, DataType dataType)
    {
        switch(encodingType)
        {
			case EncodingType::delta:
                return DeltaEncoding(dataType);
			default:
				throw NotImplementedException("No such encoding type - encoding not implemented");
        }
    }
};

} /* namespace ddj */
#endif /* DDJ_ENCODING_FACTORY_HPP_ */
