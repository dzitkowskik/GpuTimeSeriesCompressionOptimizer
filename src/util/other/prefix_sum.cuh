#include "core/cuda_ptr.hpp"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#ifndef DDJ_PREFIX_SUM_CUH_
#define DDJ_PREFIX_SUM_CUH_

namespace ddj
{

template<typename T> inline
SharedCudaPtr<T> inclusivePrefixSum_thrust(SharedCudaPtr<T> data)
{
    auto result = CudaPtr<T>::make_shared(data->size());
    if(data->get() != NULL && data->size() != 0)
    {
    	thrust::device_ptr<T> result_ptr(result->get());
    	thrust::device_ptr<T> data_ptr(data->get());
    	thrust::inclusive_scan(data_ptr, data_ptr+data->size(), result_ptr);
    }
    return result;
}

template<typename T> inline
SharedCudaPtr<T> inclusivePrefixSum_thrust(SharedCudaPtr<T> data, T& last)
{
    auto result = CudaPtr<T>::make_shared(data->size());
    if(data->get() != NULL && data->size() != 0)
    {
    	thrust::device_ptr<T> result_ptr(result->get());
    	thrust::device_ptr<T> data_ptr(data->get());
    	last = thrust::inclusive_scan(data_ptr, data_ptr+data->size(), result_ptr);
    }
    else last = 0;
    return result;
}

template<typename T> inline
SharedCudaPtr<T> exclusivePrefixSum_thrust(SharedCudaPtr<T> data)
{
    auto result = CudaPtr<T>::make_shared(data->size());
    if(data->get() != NULL && data->size() > 0)
    {
    	thrust::device_ptr<T> result_ptr(result->get());
    	thrust::device_ptr<T> data_ptr(data->get());
    	thrust::exclusive_scan(data_ptr, data_ptr+data->size(), result_ptr);
    }
    return result;
}

template<typename T> inline
SharedCudaPtr<T> exclusivePrefixSum_thrust(SharedCudaPtr<T> data, T& last)
{
    auto result = CudaPtr<T>::make_shared(data->size());
    if(data->get() != NULL && data->size() > 0)
	{
    	thrust::device_ptr<T> result_ptr(result->get());
    	thrust::device_ptr<T> data_ptr(data->get());
    	thrust::exclusive_scan(data_ptr, data_ptr+(data->size()), result_ptr);
    	last = (result_ptr+data->size()-1)[0] + (data_ptr+data->size()-1)[0];
    }
    else last = 0;
    return result;
}

} /* namespace ddj */
#endif /* DDJ_PREFIX_SUM_CUH_ */
