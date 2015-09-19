#include "core/cuda_ptr.hpp"
#include <gtest/gtest.h>
#include <boost/pointer_cast.hpp>

TEST(OtherTests, SharedCudaPtr_ReinterpretCast)
{
    const int N = 100;
    auto ptr = CudaPtr<int>::make_shared(N);
    int h_data[N];
    for(int i = 0; i < N; i++)
        h_data[i] = i;

    int* h_data_int_ptr = (int*)&h_data[0];

    ptr->fillFromHost(h_data_int_ptr, N);

    auto ptr_float = boost::reinterpret_pointer_cast<CudaPtr<float>>(ptr);
    auto h_vector = ptr_float->copyToHost();
    for(auto& item : *h_vector)
        std::cout << item << std::endl;
}
