#ifndef DDJ_COMPRESSION_THRUST_RLE_CUH_
#define DDJ_COMPRESSION_THRUST_RLE_CUH_

#define DDJ_THRUST_RLE_DEBUG 0

namespace ddj
{

class ThrustRleCompression
{
public:
    // For now I assume that data is an array of floats
    void* Encode(void* data, int in_size, int& out_size);
    void* Decode(void* data, int in_size, int& out_size);
};

} /* namespace ddj */
#endif
