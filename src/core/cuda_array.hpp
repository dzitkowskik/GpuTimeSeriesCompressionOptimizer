/*
 * cuda_array.hpp
 * 26-03-2015 Karol Dzitkowski
 */

#ifndef DDJ_CORE_CUDA_ARRAY_HPP_
#define DDJ_CORE_CUDA_ARRAY_HPP_

#include "core/cuda_ptr.hpp"
#include "core/logger.h"
#include <string>
#include <sstream>
#include <iostream>
#include <thrust/device_vector.h>

namespace ddj
{

template<typename T>
bool CompareDeviceArrays(T* a, T* b, int size);

class CudaArray
{
public:
    template<typename T>
    std::string ToString(SharedCudaPtr<T> ptr, std::string name)
    {
        std::stringstream ss;
        PrintToStream(ss, ptr, name);
        return ss.str();
    }

    template<typename T>
    std::string ToString(SharedCudaPtr<T> ptr)
    {
        std::stringstream ss;
        PrintToStream(ss, ptr);
        return ss.str();
    }

    template<typename T>
    void Print(SharedCudaPtr<T> ptr, std::string name = "")
    {
        std::cout << ToString(ptr, name);
    }

    template<typename T>
    void PrintToStream(std::ostream& stream, SharedCudaPtr<T> ptr)
    {
        thrust::device_ptr<T> data(ptr->get());
        thrust::copy(data, data + ptr->size(), std::ostream_iterator<T>(std::cout, " "));
        stream << std::endl;
    }

    template<typename T>
    void PrintToStream(std::ostream& stream, SharedCudaPtr<T> ptr, std::string name)
    {
        thrust::device_ptr<T> data(ptr->get());
        stream << name << std::endl;
        thrust::copy(data, data + ptr->size(), std::ostream_iterator<T>(std::cout, " "));
        stream << std::endl;
    }

    template<typename T>
    bool Compare(SharedCudaPtr<T> A, SharedCudaPtr<T> B)
    {
        return CompareDeviceArrays(A->get(), B->get(), A->size());
    }
};


} /* namespace ddj */
#endif /* DDJ_CORE_CUDA_ARRAY_HPP_ */
