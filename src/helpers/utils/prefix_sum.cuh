#include "core/cuda_ptr.hpp"
#include <thrust/scan.h>

namespace ddj
{

template<typename T> inline
SharedCudaPtr<T> inclusivePrefixSum_thrust(SharedCudaPtr<T> data)
{
    auto result = CudaPtr<T>::make_shared(data->size());
    thrust::inclusive_scan(data->get(), data->get()+data->size(), result->get());
    return result;
}

template<typename T> inline
SharedCudaPtr<T> inclusivePrefixSum_thrust(SharedCudaPtr<T> data, T& last)
{
    auto result = CudaPtr<T>::make_shared(data->size());
    last = thrust::inclusive_scan(data->get(), data->get()+data->size(), result->get());
    return result;
}

template<typename T> inline
SharedCudaPtr<T> exclusivePrefixSum_thrust(SharedCudaPtr<T> data)
{
    auto result = CudaPtr<T>::make_shared(data->size());
    thrust::exclusive_scan(data->get(), data->get()+data->size(), result->get());
    return result;
}

template<typename T> inline
SharedCudaPtr<T> exclusivePrefixSum_thrust(SharedCudaPtr<T> data, T& last)
{
    auto result = CudaPtr<T>::make_shared(data->size());
    last = thrust::exclusive_scan(data->get(), data->get()+data->size(), result->get());
    return result;
}

} /* namespace ddj */
