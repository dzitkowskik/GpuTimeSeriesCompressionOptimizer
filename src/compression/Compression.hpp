/*
 * Compression.hpp
 *
 *  Created on: 10-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_COMPRESSION_HPP_
#define DDJ_COMPRESSION_HPP_

namespace ddj {

class Compression
{
public:
	SharedCudaPtr<char> Encode(SharedCudaPtr<int> data);
	SharedCudaPtr<char> Encode(SharedCudaPtr<float> data);
	SharedCudaPtr<int> Decode(SharedCudaPtr<char> data);
	SharedCudaPtr<float> Decode(SharedCudaPtr<char> data);
};

} /* namespace ddj */
#endif /* DDJ_COMPRESSION_HPP_ */
