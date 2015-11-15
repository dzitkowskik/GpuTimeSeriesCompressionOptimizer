/*
 *  helper_cuda.cuh
 *
 *  Created on: 15-06-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_HELPER_CUDA_CUH_
#define DDJ_HELPER_CUDA_CUH_

#include "core/execution_policy.hpp"

namespace ddj {

template <typename KERNEL, typename... Arguments>
void cudaLaunch(ExecutionPolicy policy, KERNEL k, Arguments... args)
{
    k<<<policy.getGrid(), policy.getBlock(), policy.getShared(), policy.getStream()>>>(args...);
}

}
#endif /* DDJ_HELPER_CUDA_CUH_ */
