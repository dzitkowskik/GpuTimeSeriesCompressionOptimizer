#include "dict_compression_unittest.hpp"
#include "util/histogram/cuda_histogram.hpp"
#include "helpers/helper_print.hpp"
#include "helpers/helper_comparison.cuh"
#include "compression/dict/dict_encoding.hpp"
#include "util/splitter/splitter.hpp"

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
        if(h_mostFrequent->at(i) != h_histogramKeys->at(i))
            return false;

    return true;
}

INSTANTIATE_TEST_CASE_P(MostFreqCnt_Inst, DictCompressionTest, ::testing::Values(1, 5));

TEST_P(DictCompressionTest, GetMostFrequent_fake_data)
{
	DictEncoding dictEncoding;

    int mostFreqCnt = GetParam();

    auto fakedHistogram = fakeHistogram(size);
    auto mostFrequent = dictEncoding.GetMostFrequent(fakedHistogram, mostFreqCnt);
    int expected = size;
    int actual;
    CUDA_CALL( cudaMemcpy(&actual, mostFrequent->get(), sizeof(int), CPY_DTH) );
    EXPECT_EQ(expected, actual);
}

TEST_P(DictCompressionTest, GetMostFrequent_random_int)
{
	DictEncoding dictEncoding;
    int mostFreqCnt = GetParam();
    CudaHistogram histogram;
    auto randomHistogram = histogram.IntegerHistogram(d_int_random_data);
    auto mostFrequent = dictEncoding.GetMostFrequent(randomHistogram, mostFreqCnt);
    EXPECT_TRUE( CheckMostFrequent(randomHistogram, mostFrequent,  mostFreqCnt) );
}

TEST_P(DictCompressionTest, CompressDecompressMostFrequent_random_int)
{
    DictEncoding dictEncoding;
    CudaHistogram histogram;
    Splitter splitter;

    int mostFreqCnt = GetParam();
    auto randomHistogram = histogram.IntegerHistogram(d_int_random_data);
    auto mostFrequent = dictEncoding.GetMostFrequent(randomHistogram, mostFreqCnt);

    auto stencil = dictEncoding.GetMostFrequentStencil(d_int_random_data, mostFrequent);
    auto mostFrequentDataPart = std::get<0>(splitter.Split(d_int_random_data, stencil));

    auto encoded = dictEncoding.CompressMostFrequent(mostFrequentDataPart, mostFrequent);
    auto decoded =
        dictEncoding.DecompressMostFrequent(
            encoded,
            mostFrequent->size(),
            mostFrequentDataPart->size()
            );

    EXPECT_EQ(mostFrequentDataPart->size(), decoded->size());
    EXPECT_TRUE(
        CompareDeviceArrays(
            mostFrequentDataPart->get(),
            decoded->get(),
            mostFrequentDataPart->size())
        );
}

TEST_P(DictCompressionTest, ComressDecompress_random_int_noexception)
{
	DictEncoding encoding;
	auto compressed = encoding.Encode(d_int_random_data);
	auto decompressed = encoding.Decode(compressed);
}

TEST_P(DictCompressionTest, ComressDecompress_random_int_size)
{
	DictEncoding encoding;
	auto compressed = encoding.Encode(d_int_random_data);
	auto decompressed = encoding.Decode(compressed);

	EXPECT_EQ(d_int_random_data->size(), decompressed->size());
}

TEST_P(DictCompressionTest, ComressDecompress_random_int_data)
{
	DictEncoding encoding;
	auto data = d_int_random_data;
	auto compressed = encoding.Encode(data);
	auto decompressed = encoding.Decode(compressed);

	EXPECT_TRUE( CompareDeviceArrays(data->get(), decompressed->get(), data->size()) );
}

} /* namespace ddj */
