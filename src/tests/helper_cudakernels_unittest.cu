#include "helper_cudakernels_unittest.hpp"
#include "helpers/helper_comparison.cuh"
#include "helpers/helper_macros.h"
#include "helpers/helper_cudakernels.cuh"
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

namespace ddj
{

TEST_F(HelperCudaKernelsTest, Modulo_Kernel_simple)
{
	int N = 1000;
    thrust::device_vector<int> d_data(N);
    thrust::fill(d_data.begin(), d_data.end(), 17);
    int* d_data_raw = thrust::raw_pointer_cast(d_data.data());
    HelperCudaKernels::ModuloKernel(d_data_raw, N, 5);
    thrust::host_vector<int> h_data = d_data;
    for(auto &d : h_data)
    	EXPECT_EQ(2, d);
}

TEST_F(HelperCudaKernelsTest, Modulo_Thrust_simple)
{
	int N = 1000;
    thrust::device_vector<int> d_data(N);
    thrust::fill(d_data.begin(), d_data.end(), 17);
    int* d_data_raw = thrust::raw_pointer_cast(d_data.data());
    HelperCudaKernels::ModuloThrust(d_data_raw, N, 5);
    thrust::host_vector<int> h_data = d_data;
    for(auto &d : h_data)
    	EXPECT_EQ(2, d);
}

TEST_F(HelperCudaKernelsTest, Modulo_Kernel_complex)
{
	int N = 1000;
    thrust::device_vector<int> d_data(N);
    thrust::sequence(d_data.begin(), d_data.end(), 1, 3);
    int* d_data_raw = thrust::raw_pointer_cast(d_data.data());
    HelperCudaKernels::ModuloKernel(d_data_raw, N, 13);
    thrust::host_vector<int> h_data = d_data;
    int i = 1;
    for(auto &d : h_data)
    {
    	EXPECT_EQ(i%13, d);
    	i+=3;
    }
}

TEST_F(HelperCudaKernelsTest, Modulo_Thrust_complex)
{
	int N = 1000;
    thrust::device_vector<int> d_data(N);
    thrust::sequence(d_data.begin(), d_data.end(), 1, 3);
    int* d_data_raw = thrust::raw_pointer_cast(d_data.data());
    HelperCudaKernels::ModuloThrust(d_data_raw, N, 13);
    thrust::host_vector<int> h_data = d_data;
    int i = 1;
    for(auto &d : h_data)
    {
    	EXPECT_EQ(i%13, d);
    	i+=3;
    }
}

} /* namespace ddj */
