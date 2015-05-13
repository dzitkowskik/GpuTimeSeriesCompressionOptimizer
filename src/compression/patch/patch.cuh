/*
 *  patch.cuh
 *
 *  Created on: 13-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_COMPRESSION_PATCH_CUH_
#define DDJ_COMPRESSION_PATCH_CUH_

#include "core/cuda_ptr.hpp"
#include <tuple>

namespace ddj {

class SimplePatch
{
public:
	// template<typename T, int N>
    // boost::array<SharedCudaPtr<T>, N> Split(SharedCudaPtr<T> data);
    //
	// template<typename T> SharedCudaPtr<T> Merge(SharedCudaPtr<char> data);

    std::tuple<SharedCudaPtr<float>, SharedCudaPtr<float>> split(
        SharedCudaPtr<float> data, float low, float high);
};

} /* namespace ddj */
#endif /* DDJ_COMPRESSION_PATCH_CUH_ */
