#include "histogram_unittest.hpp"
#include "helpers/helper_comparison.cuh"
#include "helpers/helper_macros.h"
#include "core/cuda_ptr.hpp"
#include "util/histogram/basic_thrust_histogram.cuh"

#include <cuda_runtime_api.h>
#include <vector>

namespace ddj
{

TEST(SimpleCpuHistogramTest, AllOnes)
{
    int N = 1000;
    SimpleCpuHistogram histogram;
    std::vector<int> data(N);
    for(int i = 0; i < N; i++) data[i] = i;
    std::vector<int> expected(N);
    for(int i = 0; i < N; i++) expected[i] = 1;
    auto actual = histogram.Histogram(data);
    ASSERT_EQ(expected.size(), actual.size());
    for(int i = 0; i < N; i++) EXPECT_EQ(expected[i], actual[i]);
}

TEST(SimpleCpuHistogramTest, RepeatedNumbers_Modulo)
{
    int N = 1000;
    int M = 10;
    SimpleCpuHistogram histogram;
    std::vector<int> data(N);
    for(int i = 0; i < N; i++) data[i] = i%M;
    std::vector<int> expected(M);
    for(int i = 0; i < M; i++) expected[i] = N/M;
    auto actual = histogram.Histogram(data);
    ASSERT_EQ(expected.size(), actual.size());
    for(int i = 0; i < M; i++) EXPECT_EQ(expected[i], actual[i]);
}

template<typename T>
std::map<T, int> TransformToHostMap(const SharedCudaPtrPair<T, int>& d_map_data)
{
	if(d_map_data.first->size() != d_map_data.second->size())
		throw new std::runtime_error("sizes of arrays must match");
	int size = d_map_data.second->size();
	std::map<T, int> result;
	if(size > 0)
	{
		T* keys = new T[size];
		int* counts = new int[size];
		CUDA_CALL( cudaMemcpy(keys, d_map_data.first->get(), size*sizeof(T), CPY_DTH) );
		CUDA_CALL( cudaMemcpy(counts, d_map_data.second->get(), size*sizeof(int), CPY_DTH) );


		for(int i = 0; i < size; i++)
			result.insert(std::make_pair(keys[i], counts[i]));

		delete [] keys;
		delete [] counts;
	}
	return result;
}

template<typename T>
bool CompareHistograms(std::map<T, int> A, std::map<T, int> B)
{
	for(auto&& elem : A)
		if(elem.second != B[elem.first]) return false;
	return true;
}

void HistogramTest::RandomIntegerArrayTestCase(HistogramBase& histogram)
{
	int* h_data = new int[size];
	CUDA_CALL( cudaMemcpy(h_data, d_int_random_data->get(), size*sizeof(int), CPY_DTH) );
	auto h_data_vector = std::vector<int>(h_data, h_data+size);
	auto h_expected = cpu_histogram.Histogram(h_data_vector);
	auto d_actual = histogram.IntegerHistogram(d_int_random_data);
	ASSERT_TRUE(d_actual.first != NULL);
	ASSERT_TRUE(d_actual.second != NULL);
	auto h_actual = TransformToHostMap(d_actual);
	ASSERT_EQ( h_expected.size(), h_actual.size() );
	EXPECT_TRUE( CompareHistograms(h_expected, h_actual) );
	delete [] h_data;
}

TEST_F(HistogramTest, BasicThrustHistogram_RandomIntegerArray)
{
	auto histogram = BasicThrustHistogram();
	RandomIntegerArrayTestCase(histogram);
}

} /* namespace ddj */
