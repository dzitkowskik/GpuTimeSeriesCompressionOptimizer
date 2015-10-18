/*
 *  encoding_type.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_ENCODING_TYPE_HPP_
#define DDJ_ENCODING_TYPE_HPP_

namespace ddj {

enum class EncodingType {
    delta,
    // afl,
    scale,
    rle,
    dict,
    unique,
    patch,
    none,
    length
};

} /* namespace ddj */
#endif /* DDJ_ENCODING_TYPE_HPP_ */
