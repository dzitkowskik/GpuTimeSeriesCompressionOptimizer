#include "dict_compression_unittest.hpp"
#include "util/histogram/cuda_histogram.hpp"
#include "helpers/helper_print.hpp"
#include "compression/dict/dict_encoding.hpp"
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

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

bool CheckMostFrequent(
    SharedCudaPtrPair<int,int> histogram,
    SharedCudaPtr<int> mostFrequent,
    int mostFreqCnt)
{
    auto h_histogramKeys = histogram.first->copyToHost();
    auto h_histogramValues = histogram.second->copyToHost();
    auto h_mostFrequent = mostFrequent->copyToHost();

    // Sort in descending order using thrust on host using values as keys (CPU)
    thrust::sort_by_key(
        thrust::host,
        h_histogramValues->data(),
        h_histogramValues->data() + h_histogramKeys->size(),
        h_histogramKeys->data(),
        thrust::greater<int>());

    for(int i = 0; i < mostFreqCnt; i++)
    {
        if(h_mostFrequent->at(i) != h_histogramKeys->at(i))
        {
            // printf("Expected: %d, Actual: %d\n", h_histogramKeys->at(i), h_mostFrequent->at(i));
            return false;
        }
    }

    return true;
}

TEST_F(DictCompressionTest, GetMostFrequent_random_data)
{
    int mostFreqCnt = 1;
    CudaHistogram histogram;
    auto randomHistogram = histogram.IntegerHistogram(d_int_random_data);
    // HelperPrint::PrintSharedCudaPtrPair(randomHistogram, "randomHistogram");
    DictEncoding dictEncoding;
    auto mostFrequent = dictEncoding.GetMostFrequent(randomHistogram, mostFreqCnt);
    // HelperPrint::PrintArray(mostFrequent->get(), mostFrequent->size(), "mostFrequent");
    EXPECT_TRUE( CheckMostFrequent(randomHistogram, mostFrequent,  mostFreqCnt) );
}

} /* namespace ddj */
