/*
 *  simple_patch.cuh
 *
 *  Created on: 13-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_SPLITTING_SIMPLE_PATCH_CUH_
#define DDJ_SPLITTING_SIMPLE_PATCH_CUH_

#include "core/cuda_ptr.hpp"
#include <tuple>

namespace ddj {

using namespace std;

template<typename T>
class SimplePatch
{
private:
    T _low;
    T _high;

public:
    SimplePatch(T low, T high) : _low(low), _high(high) {}
    virtual ~SimplePatch();

    tuple<SharedCudaPtr<T>, SharedCudaPtr<T>> split(SharedCudaPtr<T> data);
    SharedCudaPtr<char> partition(SharedCudaPtr<T> data);
};

} /* namespace ddj */
#endif /* DDJ_SPLITTING_SIMPLE_PATCH_CUH_ */
