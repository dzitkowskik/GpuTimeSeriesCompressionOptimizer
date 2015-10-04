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

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace ddj {

class EncodingFactory
{
public:
	static boost::shared_ptr<Encoding> Get(EncodingType encodingType)
    {
        switch(encodingType)
        {
			case EncodingType::delta:
                return boost::make_shared<DeltaEncoding>();
			default:
				throw NotImplementedException("Encoding of this type not implemented");
        }
    }
};

} /* namespace ddj */
#endif /* DDJ_ENCODING_FACTORY_HPP_ */
