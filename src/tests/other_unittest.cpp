#include "core/cuda_ptr.hpp"
#include <gtest/gtest.h>
#include <boost/pointer_cast.hpp>

TEST(OtherTests, SharedCudaPtr_ReinterpretCast)
{
    const int N = 100;

    // Create fake int data
    int h_data[N];
    for(int i = 0; i < N; i++)
        h_data[i] = i;

    // Copy to device
    auto ptr = CudaPtr<int>::make_shared(N);
    ptr->fillFromHost((int*)h_data, N);
    auto expected = ptr->copyToHost();

    // Convert to float vector
    auto ptr_float = boost::reinterpret_pointer_cast<CudaPtr<float>>(ptr);
    auto h_vector_float = ptr_float->copyToHost();

    // Convert back to int vector
    auto ptr_int = boost::reinterpret_pointer_cast<CudaPtr<int>>(ptr_float);
    auto actual = ptr_int->copyToHost();

    // Compare vectors
    for(int i=0; i<expected->size(); i++)
        EXPECT_EQ((*expected)[i], (*actual)[i]);
}
