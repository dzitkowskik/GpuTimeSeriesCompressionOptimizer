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

} /* namespace ddj */
