#include "standard_param_unittest_base.hpp"
#include "util/histogram/histogram.hpp"
#include "helpers/helper_print.hpp"
#include "helpers/helper_comparison.cuh"
#include "compression/dict/dict_encoding.hpp"
#include "compression/unique/unique_encoding.hpp"
#include "util/splitter/splitter.hpp"

#include <boost/pointer_cast.hpp>

namespace ddj
{

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

class MostFrequentTest : public StandardParamTestBase {};

TEST_F(MostFrequentTest, CompressDecompressMostFrequent_random_int)
{
    int mostFreqCnt = 4;
    auto mostFrequent = Histogram().GetMostFrequent(d_int_random_data, mostFreqCnt);

    auto stencil = DictEncoding().GetMostFrequentStencil(d_int_random_data, mostFrequent);
    auto mostFrequentDataPart = std::get<0>(Splitter().Split(d_int_random_data, stencil));

    auto encoded = UniqueEncoding().CompressUnique(mostFrequentDataPart, mostFrequent);
    auto decoded = UniqueEncoding().template DecompressUnique<int>(encoded);

    EXPECT_EQ(mostFrequentDataPart->size(), decoded->size());
    EXPECT_TRUE(
        CompareDeviceArrays(
            mostFrequentDataPart->get(),
            decoded->get(),
            mostFrequentDataPart->size())
        );
}



}
