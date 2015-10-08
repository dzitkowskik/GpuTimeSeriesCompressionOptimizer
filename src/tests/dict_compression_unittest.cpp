#include "dict_compression_unittest.hpp"
#include "util/histogram/histogram.hpp"
#include "helpers/helper_print.hpp"
#include "helpers/helper_comparison.cuh"
#include "compression/dict/dict_encoding.hpp"
#include "compression/unique/unique_encoding.hpp"
#include "util/splitter/splitter.hpp"

namespace ddj
{

INSTANTIATE_TEST_CASE_P(MostFreqCnt_Inst, DictCompressionTest, ::testing::Values(1, 5));

TEST_P(DictCompressionTest, CompressDecompressMostFrequent_random_int)
{
    int mostFreqCnt = GetParam();
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

//TEST_P(DictCompressionTest, ComressDecompress_random_int_noexception)
//{
//	DictEncoding encoding;
//	auto compressed = encoding.Encode(d_int_random_data);
//	auto decompressed = encoding.Decode(compressed);
//}
//
//TEST_P(DictCompressionTest, ComressDecompress_random_int_size)
//{
//	DictEncoding encoding;
//	auto compressed = encoding.Encode(d_int_random_data);
//	auto decompressed = encoding.Decode(compressed);
//
//	EXPECT_EQ(d_int_random_data->size(), decompressed->size());
//}
//
//TEST_P(DictCompressionTest, ComressDecompress_random_int_data)
//{
//	DictEncoding encoding;
//	auto data = d_int_random_data;
//	auto compressed = encoding.Encode(data);
//	auto decompressed = encoding.Decode(compressed);
//
//	EXPECT_TRUE( CompareDeviceArrays(data->get(), decompressed->get(), data->size()) );
//}

} /* namespace ddj */
