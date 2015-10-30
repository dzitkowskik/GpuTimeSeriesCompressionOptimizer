#include "helpers/helper_comparison.cuh"
#include "core/macros.h"
#include "core/cuda_ptr.hpp"
#include "util/other/cpu_histogram.hpp"
#include "util/histogram/histogram.hpp"
#include "core/config.hpp"
#include "test/unittest_base.hpp"
#include "compression/delta/delta_encoding.hpp"
#include "helpers/helper_print.hpp"

#include <cuda_runtime_api.h>
#include <vector>
#include <iostream>
#include <boost/bind.hpp>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace ddj
{

class HistogramTest : public UnittestBase {};


template<typename T>
bool CheckMostFrequent(
    SharedCudaPtrPair<T,int> histogram,
    SharedCudaPtr<T> mostFrequent,
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

TEST_F(HistogramTest, GetMostFrequent_fake_data)
{
    int mostFreqCnt = 3;
    auto fakeData = GetFakeIntDataForHistogram();
    auto fakedHistogram = Histogram().CalculateSparse(fakeData);
    auto mostFrequent = Histogram().GetMostFrequent(fakedHistogram, mostFreqCnt);
    int expected = GetSize(), actual;
    CUDA_CALL( cudaMemcpy(&actual, mostFrequent->get(), sizeof(int), CPY_DTH) );
    EXPECT_EQ(expected, actual);
}

TEST_F(HistogramTest, GetMostFrequent_random_int)
{
    int mostFreqCnt = 4;
    auto randomHistogram = Histogram().CalculateDense(GetIntRandomData());
    auto mostFrequent = Histogram().GetMostFrequent(randomHistogram, mostFreqCnt);
    EXPECT_TRUE( CheckMostFrequent(randomHistogram, mostFrequent,  mostFreqCnt) );
}

TEST(CpuHistogramTest, Sparse_AllOnes)
{
    int N = 1000;
    CpuHistogramSparse histogram;
    std::vector<int> data(N);
    for(int i = 0; i < N; i++) data[i] = i;
    std::vector<int> expected(N);
    for(int i = 0; i < N; i++) expected[i] = 1;
    auto actual = histogram.Histogram(data);
    ASSERT_EQ(expected.size(), actual.size());
    for(int i = 0; i < N; i++) EXPECT_EQ(expected[i], actual[i]);
}

TEST(CpuHistogramTest, Dense_AllOnes)
{
    int N = 1000;
    CpuHistogramDense histogram;
    std::vector<int> data(N);
    for(int i = 0; i < N; i++) data[i] = i;
    std::vector<int> expected(N);
    for(int i = 0; i < N; i++) expected[i] = 1;
    auto actual = histogram.Histogram(data);
    ASSERT_EQ(expected.size(), actual.size());
    for(int i = 0; i < N; i++) EXPECT_EQ(expected[i], actual[i]);
}

TEST(CpuHistogramTest, Sparse_RepeatedNumbers_Modulo)
{
    int N = 1000;
    int M = 10;
    CpuHistogramSparse histogram;
    std::vector<int> data(N);
    for(int i = 0; i < N; i++) data[i] = i%M;
    std::vector<int> expected(M);
    for(int i = 0; i < M; i++) expected[i] = N/M;
    auto actual = histogram.Histogram(data);
    ASSERT_EQ(expected.size(), actual.size());
    for(int i = 0; i < M; i++) EXPECT_EQ(expected[i], actual[i]);
}

TEST(CpuHistogramTest, Dense_RepeatedNumbers_Modulo)
{
    int N = 1000;
    int M = 10;
    CpuHistogramDense histogram;
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

template<typename T>
void PrintHostHistogram(std::map<T, int> histogram, std::string name)
{
    std::cout << name << std::endl;
    for(auto&& elem : histogram)
    {
        std::cout << "(" << elem.first << "," << elem.second << ")";
    }
    std::cout << std::endl;
}

template<typename T, class CPU_HIST>
void CheckHistogramResult(SharedCudaPtr<T> data, SharedCudaPtrPair<T, int> result)
{
	int size = data->size();
	CPU_HIST cpuHistogram;

	T* h_data = new T[size];
	CUDA_CALL( cudaMemcpy(h_data, data->get(), size*sizeof(T), CPY_DTH) );
	auto h_data_vector = std::vector<T>(h_data, h_data+size);
	auto h_expected = cpuHistogram.Histogram(h_data_vector);
	auto d_actual = result;
	ASSERT_TRUE(d_actual.first != NULL);
	ASSERT_TRUE(d_actual.second != NULL);
	auto h_actual = TransformToHostMap(d_actual);

	EXPECT_EQ( h_expected.size(), h_actual.size() );
	EXPECT_TRUE( CompareHistograms(h_expected, h_actual) );

//	    PrintHostHistogram(h_expected, "Expected");
//	    PrintHostHistogram(h_actual, "Actual");

	delete [] h_data;
}

TEST_F(HistogramTest, ThrustSparseHistogram_RandomIntegerArray)
{
	auto randomData = GetIntRandomData();
	auto result = Histogram().ThrustSparseHistogram(randomData);
	CheckHistogramResult<int, CpuHistogramSparse>(randomData, result);
}

TEST_F(HistogramTest, ThrustSparseHistogram_RealData_Delta_Time)
{
	auto realData = GetTsIntDataFromTestFile();
	auto deltaEncoded = DeltaEncoding().Encode(realData);
	auto realDataDelta = MoveSharedCudaPtr<char, time_t>(deltaEncoded[1]);

	auto result = Histogram().ThrustSparseHistogram(realDataDelta);
	CheckHistogramResult<time_t, CpuHistogramSparse>(realDataDelta, result);
}

TEST_F(HistogramTest, CalculateHistogram_RealData_Delta_Time)
{
	auto realData = GetTsIntDataFromTestFile();
	auto deltaEncoded = DeltaEncoding().Encode(realData);
	auto realDataDelta = MoveSharedCudaPtr<char, time_t>(deltaEncoded[1]);
	auto result = Histogram().CalculateDense(realDataDelta);

	CheckHistogramResult<time_t, CpuHistogramDense>(realDataDelta, result);
}

} /* namespace ddj */
