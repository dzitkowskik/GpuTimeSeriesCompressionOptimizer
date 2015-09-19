/*
 *  encoding_factory.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_ENCODING_FACTORY_HPP_
#define DDJ_ENCODING_FACTORY_HPP_

namespace ddj {

#include "encoding.hpp"
#include "encoding_type.hpp"
#include "data_type.hpp"
#include "compression/delta/delta_encoding.hpp"

class EncodingFactory
{
public:
	static Encoding Get(EncodingType encodingType, DataType dataType)
    {
        switch(encodingType)
        {
            case EncodingType.delta:
                return DeltaEncoding(dataType);
            case default:
                throw std::runtime_error("No encoding for this encoding type found");
        }
    }
};

} /* namespace ddj */
#endif /* DDJ_ENCODING_RESULT_HPP_ */
