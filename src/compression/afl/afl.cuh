#ifndef DDJ_COMPRESSION_AFL_CUH_
#define DDJ_COMPRESSION_AFL_CUH_

#include "../../core/logger.h"
#include <boost/noncopyable.hpp>

namespace ddj
{

class AFLCompressionMetadata
{
public:
	int min_bit;
};

class AFLCompression : private boost::noncopyable
{
private:
    Logger _logger = Logger::getInstance(LOG4CPLUS_TEXT("AFLCompression"));
    unsigned long max_size;

public:
	AFLCompression(){ max_size = 10000; }
	~AFLCompression(){}
    void* Encode(int* data, int in_size, int& out_size, AFLCompressionMetadata& metadata);
    void* Decode(void* data, int in_size, int& out_size, AFLCompressionMetadata metadata);

private:
    int getMinBitCnt(int* data, int size);
};

} /* namespace ddj */
#endif
