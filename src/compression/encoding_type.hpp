/*
 *  encoding_type.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_ENCODING_TYPE_HPP_
#define DDJ_ENCODING_TYPE_HPP_

#include "core/not_implemented_exception.hpp"

#include <string>

namespace ddj {

enum class EncodingType {
    delta,
    afl,
    scale,
    rle,
    dict,
    unique,
    patch,
    none,
    floatToInt,
    constData,
    length
};

inline std::string GetEncodingTypeString(EncodingType type)
{
	switch(type)
	{
		case EncodingType::afl: return "afl";
		case EncodingType::constData: return "constData";
		case EncodingType::delta: return "delta";
		case EncodingType::dict: return "dict";
		case EncodingType::floatToInt: return "floatToInt";
		case EncodingType::none: return "none";
		case EncodingType::patch: return "patch";
		case EncodingType::rle: return "rle";
		case EncodingType::scale: return "scale";
		case EncodingType::unique: return "unique";
		default: throw NotImplementedException("This encoding type is not implemented");
	}
}

} /* namespace ddj */
#endif /* DDJ_ENCODING_TYPE_HPP_ */
