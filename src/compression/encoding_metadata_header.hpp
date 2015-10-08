/*
 *  encoding_metadata_header.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_ENCODING_METADATA_HEADER_HPP_
#define DDJ_ENCODING_METADATA_HEADER_HPP_

namespace ddj {

struct EncodingMetadataHeader {
    int16_t EncodingType;
    int16_t DataType;
    int32_t MetadataLength;
};

} /* namespace ddj */
#endif /* DDJ_ENCODING_METADATA_HEADER_HPP_ */
