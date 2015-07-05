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
    ASSERT_EQ(expected, actual);
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
    ASSERT_EQ(expected, actual);
    printf("OK\n");
    for(int i = 0; i < M; i++)
    {
        printf("OK %d\n", i);
        EXPECT_EQ(expected[i], actual[i]); 
    }
    printf("DONE\n");
}

// void HistogramTest::RandomIntegerArrayTestCase(HistogramBase& histogram)
// {
//     int* h_data = new int[size];
//     CUDA_CALL( cudaMemcpy(h_data, d_int_random_data->get(), size*sizeof(int), CPY_DTH) );
//     auto h_data_vector = std::vector<int>(h_data, h_data+size);
//     auto serial_alg_answer = cpu_histogram.Histogram(h_data_vector);
//     ScopedCudaPtr<int> expected(new CudaPtr<int>());
//     expected->fillFromHost(serial_alg_answer.data(), serial_alg_answer.size());
//     auto actual = histogram.IntegerHistogram(d_int_random_data);
//     EXPECT_EQ( expected->size(), actual->size() );
//     EXPECT_TRUE( CompareDeviceArrays(expected->get(), actual->get(), expected->size()) );
//     delete [] h_data;
// }
//
// TEST_F(HistogramTest, BasicThrustHistogram_RandomIntegerArray)
// {
//     auto histogram = BasicThrustHistogram();
//     RandomIntegerArrayTestCase(histogram);
// }

} /* namespace ddj */
