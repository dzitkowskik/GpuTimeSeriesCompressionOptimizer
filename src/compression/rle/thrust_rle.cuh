#ifndef DDJ_COMPRESSION_THRUST_RLE_CUH_
#define DDJ_COMPRESSION_THRUST_RLE_CUH_

#include "../../core/logger.h"

#define DDJ_THRUST_RLE_DEBUG 0

namespace ddj {

class ThrustRleCompression
{
public:
    // For now I assume that data is an array of floats
	template<typename T>
    void* Encode(T* data, int in_size, int& out_size);
	template<typename T>
    T* Decode(void* data, int in_size, int& out_size);
private:
    Logger _logger = Logger::getInstance(LOG4CPLUS_TEXT("ThrustRleCompression"));
};

} /* namespace ddj */
#endif
