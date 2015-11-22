#include "core/cuda_ptr.hpp"
#include "util/copy/cuda_array_copy.hpp"
#include "test/unittest_base.hpp"

namespace ddj
{

class CudaPtrTest : public UnittestBase {};

TEST_F(CudaPtrTest, Concatenate_EqualSize_Vectors_Int_Size)
{
    int N = 10;
    SharedCudaPtrVector<int> randomIntDataVector;
    for(int i = 0; i < N; i++)
	   randomIntDataVector.push_back( GetIntRandomData() );

    size_t expected = N * GetSize();
    size_t actual = CudaArrayCopy().Concatenate(randomIntDataVector)->size();
    EXPECT_EQ(expected, actual);
}

}
