/*
 * scale.cuh
 *
 *  Created on: 24-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef SCALE_CUH_
#define SCALE_CUH_

namespace ddj
{

template<typename T> T* scaleEncode(T* data, int size, T& min);
template<typename T> T* scaleDecode(T* data, int size, T& min);

} /* namespace ddj */
#endif /* SCALE_CUH_ */
