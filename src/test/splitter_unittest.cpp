#include "splitter_unittest.hpp"
#include "helpers/helper_comparison.cuh"
#include "core/macros.h"
#include "core/cuda_ptr.hpp"
#include "util/splitter/splitter.hpp"

#include <cuda_runtime_api.h>

namespace ddj
{

TEST_F(SplitterTest, Split_zero_size)
{
    Splitter splitter;
    auto data = CudaPtr<int>::make_shared((size_t)0);
    auto stencil = CudaPtr<int>::make_shared((size_t)0);
    auto result = splitter.Split(data, stencil);

    EXPECT_EQ(0, get<0>(result)->size());
    EXPECT_EQ(0, get<1>(result)->size());
    EXPECT_EQ(NULL, get<0>(result)->get());
    EXPECT_EQ(NULL, get<1>(result)->get());
}

TEST_F(SplitterTest, Split_yes_empty)
{
    Splitter splitter;
	auto data = d_int_random_data;
	int h_stencil[size];
	auto d_stencil = CudaPtr<int>::make_shared(size);
	for(int i=0; i<size; i++)
		h_stencil[i] = 0;
	CUDA_CALL( cudaMemcpy(d_stencil->get(), h_stencil, size*sizeof(int), CPY_HTD) );
	auto result = splitter.Split(data, d_stencil);

    EXPECT_EQ(0, get<0>(result)->size());
    EXPECT_EQ(size, get<1>(result)->size());
    EXPECT_TRUE( CompareDeviceArrays(d_int_random_data->get(), get<1>(result)->get(), size) );
}

TEST_F(SplitterTest, Split_no_empty)
{
    Splitter splitter;
	auto data = d_int_random_data;
	int h_stencil[size];
	auto d_stencil = CudaPtr<int>::make_shared(size);
	for(int i=0; i<size; i++)
		h_stencil[i] = 1;
	CUDA_CALL( cudaMemcpy(d_stencil->get(), h_stencil, size*sizeof(int), CPY_HTD) );
	auto result = splitter.Split(data, d_stencil);

    EXPECT_EQ(size, get<0>(result)->size());
    EXPECT_EQ(0, get<1>(result)->size());
    EXPECT_TRUE( CompareDeviceArrays(d_int_random_data->get(), get<0>(result)->get(), size) );
}

TEST_F(SplitterTest, Split_manual_simple_int)
{
    Splitter splitter;
	auto data = d_int_random_data;
	int h_stencil[size];
	auto d_stencil = CudaPtr<int>::make_shared(size);
	for(int i=0; i<size; i++)
		h_stencil[i] = i < size/2 ? 1 : 0;
	CUDA_CALL( cudaMemcpy(d_stencil->get(), h_stencil, size*sizeof(int), CPY_HTD) );
	auto result = splitter.Split(data, d_stencil);

	EXPECT_EQ(size/2, get<0>(result)->size());
	EXPECT_EQ(size/2, get<1>(result)->size());
	EXPECT_TRUE( CompareDeviceArrays(
			d_int_random_data->get(),
			get<0>(result)->get(),
			size/2) );
	EXPECT_TRUE( CompareDeviceArrays(
			d_int_random_data->get()+(size/2),
			get<1>(result)->get(),
			size/2) );
}

TEST_F(SplitterTest, Split_manual_complex_float)
{
    Splitter splitter;
	auto data = d_float_random_data;
	int h_stencil[size];
	auto d_stencil = CudaPtr<int>::make_shared(size);
	for(int i=0; i<size; i++)
	{
		if(i < size/4) h_stencil[i] = 0;
		else if(i < size/2) h_stencil[i] = 1;
		else if(i < 3*size/4) h_stencil[i] = 0;
		else h_stencil[i] = 1;
	}
	CUDA_CALL( cudaMemcpy(d_stencil->get(), h_stencil, size*sizeof(int), CPY_HTD) );
	auto result = splitter.Split(data, d_stencil);

	EXPECT_EQ(size/2, get<0>(result)->size());
	EXPECT_EQ(size/2, get<1>(result)->size());
	EXPECT_TRUE( CompareDeviceArrays(
			d_float_random_data->get(),
			get<1>(result)->get(),
			size/4) );
	EXPECT_TRUE( CompareDeviceArrays(
			d_float_random_data->get()+(size/4),
			get<0>(result)->get(),
			size/4) );
	EXPECT_TRUE( CompareDeviceArrays(
			d_float_random_data->get()+(size/2),
			get<1>(result)->get()+(size/4),
			size/4) );
	EXPECT_TRUE( CompareDeviceArrays(
			d_float_random_data->get()+(3*size/4),
			get<0>(result)->get()+(size/4),
			size/4) );
}

TEST_F(SplitterTest, Merge_manual_complex_int)
{
    Splitter splitter;
	auto data = d_int_random_data;
	int h_stencil[size];
	auto d_stencil = CudaPtr<int>::make_shared(size);
	for(int i=0; i<size; i++)
	{
		if(i < size/4) h_stencil[i] = 0;
		else if(i < size/2) h_stencil[i] = 1;
		else if(i < 3*size/4) h_stencil[i] = 0;
		else h_stencil[i] = 1;
	}
	CUDA_CALL( cudaMemcpy(d_stencil->get(), h_stencil, size*sizeof(int), CPY_HTD) );
	auto splittedData = splitter.Split(data, d_stencil);
	auto result = splitter.Merge(splittedData, d_stencil);

	EXPECT_TRUE( CompareDeviceArrays(data->get(), result->get(), data->size()) );
}

} /* namespace ddj */
