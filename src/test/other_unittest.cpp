#include "unittest_base.hpp"
#include "util/histogram/histogram.hpp"
#include "helpers/helper_print.hpp"
#include "helpers/helper_comparison.cuh"
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

TEST(OtherTests, GetFloatPrecision_Integer_Small)
{
	float number = 123.0f;
	int expected = 0;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

TEST(OtherTests, GetFloatPrecision_Integer_Big)
{
	float number = 123123123123;
	int expected = 0;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

TEST(OtherTests, GetFloatPrecision_IntegerNegative)
{
	float number = -123123123;
	int expected = 0;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

TEST(OtherTests, GetFloatPrecision_Float_Small_1)
{
	float number = 123.1;
	int expected = 1;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

TEST(OtherTests, GetFloatPrecision_Float_Small_2)
{
	float number = 123.34;
	int expected = 2;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

TEST(OtherTests, GetFloatPrecision_Float_Small_7)
{
	float number = 13.3412312;
	int expected = 6;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

TEST(OtherTests, GetFloatPrecision_Float_Big_4)
{
	float number = 1231201.2312;
	int expected = 1;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

TEST(OtherTests, GetFloatPrecision_Float_Big_8)
{
	float number = 12312311.33412312;
	int expected = 0;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

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
