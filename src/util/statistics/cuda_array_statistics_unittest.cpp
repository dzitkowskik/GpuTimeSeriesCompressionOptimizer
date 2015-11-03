/*
 *  cuda_array_statistics_unittest.cpp
 *
 *  Created on: 10-05-2015
 *      Author: Karol Dzitkowski
 */

#include "cuda_array_statistics.hpp"
#include "util/transform/cuda_array_transform.hpp"
#include "test/unittest_base.hpp"
#include "helpers/helper_comparison.cuh"
#include "helpers/helper_print.hpp"

#include <gtest/gtest.h>
#include <cmath>

namespace ddj {

class CudaArrayStatisticsTest : public UnittestBase {};

TEST_F(CudaArrayStatisticsTest, GetFloatPrecision_Integer_Small)
{
	float number = 123.0f;
	int expected = 0;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

TEST_F(CudaArrayStatisticsTest, GetFloatPrecision_Integer_Big)
{
	float number = 123123123123;
	int expected = 0;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

TEST_F(CudaArrayStatisticsTest, GetFloatPrecision_IntegerNegative)
{
	float number = -123123123;
	int expected = 0;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

TEST_F(CudaArrayStatisticsTest, GetFloatPrecision_Float_Small_1)
{
	float number = 123.1;
	int expected = 1;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

TEST_F(CudaArrayStatisticsTest, GetFloatPrecision_Float_Small_2)
{
	float number = 123.34;
	int expected = 2;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

TEST_F(CudaArrayStatisticsTest, GetFloatPrecision_Float_Small_7)
{
	float number = 13.3412312;
	int expected = 6;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

TEST_F(CudaArrayStatisticsTest, GetFloatPrecision_Float_Big_4)
{
	float number = 1231201.2312;
	int expected = 1;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

TEST_F(CudaArrayStatisticsTest, GetFloatPrecision_Float_Big_8)
{
	float number = 12312311.33412312;
	int expected = 0;
	int actual = _getFloatPrecision(number);
	EXPECT_EQ(expected, actual);
}

TEST_F(CudaArrayStatisticsTest, Precision_RandomFloats_Precision_3)
{
	auto randomData = GetFloatRandomDataWithMaxPrecision(3);
	int precision = CudaArrayStatistics().Precision(randomData);

//	printf("Precision = %d\n", precision);

	FloatingPointToIntegerOperator<float, int> opMul { precision };
	SharedCudaPtr<int> transformed = CudaArrayTransform()
		.Transform<float, int>(randomData, opMul);

	IntegerToFloatingPointOperator<int, float> opDiv { precision };
	SharedCudaPtr<float> transformedBack = CudaArrayTransform()
		.Transform<int, float>(transformed, opDiv);

//	auto a = randomData->copyToHost();
//	auto b = transformed->copyToHost();
//	auto c = transformedBack->copyToHost();
//
//	for(int i = 0; i < a->size(); i++)
//		if((*a)[i] != (*c)[i])
//			printf("ERROR: a = %f, b = %d, c = %f\n", (*a)[i], (*b)[i], (*c)[i]);

	EXPECT_TRUE(
		CompareDeviceArrays(randomData->get(), transformedBack->get(), randomData->size())
			);
}

} /* namespace ddj */
