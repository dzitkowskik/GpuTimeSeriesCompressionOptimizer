#include "dict_compression_unittest.hpp"
#include "util/histogram/cuda_histogram.hpp"
#include "helpers/helper_print.hpp"
#include "compression/dict/dict_encoding.hpp"

namespace ddj
{

SharedCudaPtrPair<int, int> fakeHistogram(int size)
{
    int mod = size / 10;
    int big = size/3;
    std::vector<int> h_fakeData;
    for(int i = 0; i < size; i++)
    {
        if(i%mod == 0 || i%mod == 1)
            h_fakeData.push_back(size);
        else
            h_fakeData.push_back(i%mod);
    }
    auto d_fakeData = CudaPtr<int>::make_shared(size);
    d_fakeData->fillFromHost(h_fakeData.data(), size);
    CudaHistogram histogram;
    return histogram.IntegerHistogram(d_fakeData);
}


TEST_F(DictCompressionTest, GetMostFrequent_fake_data)
{
    auto fakedHistogram = fakeHistogram(size);
    // HelperPrint::PrintSharedCudaPtrPair(fakedHistogram, "fakedHistogram");
    DictEncoding dictEncoding;
    auto mostFrequent = dictEncoding.GetMostFrequent(fakedHistogram, 1);
    // HelperPrint::PrintArray(mostFrequent->get(), mostFrequent->size(), "mostFrequent");
    int expected = size;
    int actual;
    CUDA_CALL( cudaMemcpy(&actual, mostFrequent->get(), sizeof(int), CPY_DTH) );
    EXPECT_EQ(expected, actual);
}

} /* namespace ddj */
