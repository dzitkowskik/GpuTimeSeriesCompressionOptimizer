#include "unittest_base.hpp"
#include "util/histogram/histogram.hpp"

#include "core/cuda_array.hpp"
#include "compression/dict/dict_encoding.hpp"
#include "compression/unique/unique_encoding.hpp"
#include "util/splitter/splitter.hpp"
#include "util/statistics/cuda_array_statistics.hpp"

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

// class ANyseTest : public UnittestBase {};
//
// TEST_F(ANyseTest, CreateNyseUnittestFileInSampleData)
// {
// 	Save1MFrom1GNyseDataInSampleData(1e5);
// }

class MostFrequentTest : public UnittestBase {};

TEST_F(MostFrequentTest, CompressDecompressMostFrequent_random_int)
{
	auto randomData = GetIntRandomData();
    int mostFreqCnt = 4;
    auto mostFrequent = Histogram().GetMostFrequent(randomData, mostFreqCnt);

    auto stencil = DictEncoding().GetMostFrequentStencil(randomData, mostFrequent);
    auto mostFrequentDataPart = std::get<0>(Splitter().Split(randomData, stencil));

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
