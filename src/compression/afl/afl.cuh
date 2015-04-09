#ifndef DDJ_COMPRESSION_AFL_CUH_
#define DDJ_COMPRESSION_AFL_CUH_

#include "../../core/logger.h"
#include <boost/noncopyable.hpp>

namespace ddj
{

class AFLCompression : private boost::noncopyable
{
public:
	AFLCompression(){ max_size = 10000; }
	~AFLCompression(){}
    void* Encode(int* data, int in_size, int& out_size);
    void* Decode(int* data, int in_size, int& out_size);
private:
    Logger _logger = Logger::getInstance(LOG4CPLUS_TEXT("AFLCompression"));
    unsigned long max_size;
};

} /* namespace ddj */
#endif
