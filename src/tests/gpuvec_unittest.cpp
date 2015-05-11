#include "gpuvec_unittest.hpp"
#include "helpers/helper_generator.hpp"
#include "helpers/helper_comparison.cuh"

namespace ddj {

// check thrust version
TEST_F(GpuVecTest, GpuVec_Constructor_ZeroSize)
{
	GpuVec vec;
	EXPECT_EQ(0, vec.Size());
}

TEST_F(GpuVecTest, GpuVec_Constructor_NonZeroSize)
{
	GpuVec vec(10);
	EXPECT_EQ(10, vec.Size());
}

TEST_F(GpuVecTest, GpuVec_Constructor_WriteRead)
{
	GpuVec vec;
	HelperGenerator gen;
	float* d_data = gen.GenerateRandomFloatDeviceArray(10);

	auto offset = vec.Write(d_data, 10*sizeof(float));
	float* d_actual = (float*)vec.Read(offset, 10*sizeof(float));
	EXPECT_TRUE(CompareDeviceArrays(d_data, d_actual, 10));

	CUDA_CALL(cudaFree(d_data));
}

} /* namespace ddj */
