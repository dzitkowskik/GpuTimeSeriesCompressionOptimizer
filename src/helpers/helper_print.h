/*
 * helper_print.h 26-03-2015 Karol Dzitkowski
 */
#ifndef DDJ_HELPER_PRINT_H_
#define DDJ_HELPER_PRINT_H_

#include <thrust/device_vector.h>
#include <iostream>

class HelperPrint
{
public:
    template <typename T> static void PrintDeviceVector(
        thrust::device_vector<T> data, const char* name)
    {
        std::cout << name << std::endl;
        thrust::copy(data.begin(), data.end(),
            std::ostream_iterator<float>(std::cout, ""));
        std::cout << std::endl;
    }

    template <typename T> static void PrintDevicePtr(
        thrust::device_ptr<T> data, int size, const char* name)
    {
        std::cout << name << std::endl;
        thrust::copy(data, data+size,
            std::ostream_iterator<float>(std::cout, ""));
        std::cout << std::endl;
    }

    template <typename R, typename L>
    static void PrintDeviceVectors(thrust::device_vector<R> a,
        thrust::device_vector<L> b, const char* name)
    {
        std::cout << name << std::endl;
        for(size_t i = 0; i < a.size(); i++)
            std::cout << "(" << a[i] << "," << b[i] << ")";
        std::cout << std::endl;
    }
};

#endif /* DDJ_HELPER_PRINT_H_ */
