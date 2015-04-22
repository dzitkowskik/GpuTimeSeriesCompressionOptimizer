#ifndef DDJ_COMPRESSION_THRUST_RLE_CUH_
#define DDJ_COMPRESSION_THRUST_RLE_CUH_

#define DDJ_THRUST_RLE_DEBUG 0

namespace ddj
{

class ThrustRleCompression
{
public:
	template<typename T>
    void* Encode(T* data, int in_size, int& out_size);
	template<typename T>
    T* Decode(void* data, int in_size, int& out_size);
};

} /* namespace ddj */
#endif
