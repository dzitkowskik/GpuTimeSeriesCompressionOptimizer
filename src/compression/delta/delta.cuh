/*
 * delta.cuh
 *
 *  Created on: 18-04-2015
 *      Author: ghash
 */

#ifndef DELTA_CUH_
#define DELTA_CUH_

namespace ddj
{

template<typename T> T* deltaEncode(T* data, int size, T& first);
template<typename T> T deltaEncodeInPlace(T* data, int size);
template<typename T> T* deltaDecode(T* data, int size, T first);
template<typename T> void deltaDecodeInPlace(T* data, int size, T first);

} /* namespace ddj */
#endif /* DELTA_CUH_ */
