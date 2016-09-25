#include "core/cuda_ptr.hpp"
#include "core/cuda_device.hpp"
#include "util/copy/cuda_array_copy.hpp"
#include "test/unittest_base.hpp"

namespace ddj
{

class CudaPtrTest : public UnittestBase {};

TEST_F(CudaPtrTest, Concatenate_EqualSize_Vectors_Int_Size)
{
    CudaDevice hc;
    int devId = hc.SetCudaDeviceWithMaxFreeMem();
    cudaGetLastError();

    int N = 10;
    SharedCudaPtrVector<int> randomIntDataVector;
    for(int i = 0; i < N; i++)
	   randomIntDataVector.push_back( GetIntRandomData() );

    size_t expected = N * GetSize();
    size_t actual = CudaArrayCopy().Concatenate(randomIntDataVector)->size();
    EXPECT_EQ(expected, actual);
}

TEST_F(CudaPtrTest, MakeShared_AllocateArray_Size_32M_Int)
{
    CudaDevice hc;
    int devId = hc.SetCudaDeviceWithMaxFreeMem();
    cudaGetLastError();

    int N = 1<<20;
    auto data = CudaPtr<int>::make_shared(N);
    data->set(1);
    data->clear();
}

// TEST_F(CudaPtrTest, MakeShared_AllocateArray_Size_64M_Int)
// {
//     int N = 1<<26;
//     auto data = CudaPtr<int>::make_shared(N);
//     data->set(1);
//     data->clear();
// }

// TEST_F(CudaPtrTest, MakeShared_AllocateArray_Size_128M_Int)
// {
//     int N = 1<<27;
//     auto data = CudaPtr<int>::make_shared(N);
//     data->set(1);
//     data->clear();
// }

}
