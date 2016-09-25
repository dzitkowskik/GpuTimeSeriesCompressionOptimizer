/*
 *  encoding_factory.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_ENCODING_FACTORY_HPP_
#define DDJ_ENCODING_FACTORY_HPP_

#include "compression/encoding.hpp"
#include "compression/encoding_type.hpp"
#include "data/data_type.hpp"
#include "core/cuda_ptr.hpp"
#include "core/not_implemented_exception.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace ddj {

class EncodingFactory
{
public:
	DataType dataType;
	EncodingType encodingType;

public:
	EncodingFactory(DataType dt, EncodingType et)
		: dataType(dt), encodingType(et)
	{}
	virtual ~EncodingFactory(){}
	EncodingFactory(const EncodingFactory& other)
		: dataType(other.dataType), encodingType(other.encodingType)
	{}

public:
	virtual boost::shared_ptr<Encoding> Get() = 0;
	virtual boost::shared_ptr<Encoding> Get(SharedCudaPtr<char> data) = 0;
};

} /* namespace ddj */
#endif /* DDJ_ENCODING_FACTORY_HPP_ */
