#include "core/cuda_ptr.hpp"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

namespace ddj
{

template<typename T> inline
SharedCudaPtr<T> inclusivePrefixSum_thrust(SharedCudaPtr<T> data)
{
    auto result = CudaPtr<T>::make_shared(data->size());
    thrust::device_ptr<T> result_ptr(result->get());
    thrust::device_ptr<T> data_ptr(data->get());
    if(data->get() != NULL && data->size() != 0)
    	thrust::inclusive_scan(data_ptr, data_ptr+data->size(), result_ptr);
    return result;
}

template<typename T> inline
SharedCudaPtr<T> inclusivePrefixSum_thrust(SharedCudaPtr<T> data, T& last)
{
    auto result = CudaPtr<T>::make_shared(data->size());
    thrust::device_ptr<T> result_ptr(result->get());
    thrust::device_ptr<T> data_ptr(data->get());
    if(data->get() != NULL && data->size() != 0)
    	last = thrust::inclusive_scan(data_ptr, data_ptr+data->size(), result_ptr);
    else last = 0;
    return result;
}

template<typename T> inline
SharedCudaPtr<T> exclusivePrefixSum_thrust(SharedCudaPtr<T> data)
{
    auto result = CudaPtr<T>::make_shared(data->size());
    thrust::device_ptr<T> result_ptr(result->get());
    thrust::device_ptr<T> data_ptr(data->get());
    if(data->get() != NULL && data->size() != 0)
    	thrust::exclusive_scan(data_ptr, data_ptr+data->size(), result_ptr);
    return result;
}

template<typename T> inline
SharedCudaPtr<T> exclusivePrefixSum_thrust(SharedCudaPtr<T> data, T& last)
{
    auto result = CudaPtr<T>::make_shared(data->size());
    thrust::device_ptr<T> result_ptr(result->get());
    thrust::device_ptr<T> data_ptr(data->get());
    if(data->get() != NULL && data->size() != 0)
    {
    	thrust::exclusive_scan(data_ptr, data_ptr+(data->size()), result_ptr);
    	last = (result_ptr+data->size()-1)[0] + (data_ptr+data->size()-1)[0];
    }
    else last = 0;
    return result;
}

} /* namespace ddj */
