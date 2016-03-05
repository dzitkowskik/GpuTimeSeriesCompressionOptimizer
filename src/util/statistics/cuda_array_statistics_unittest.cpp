/*
 *  cuda_array_statistics_unittest.cpp
 *
 *  Created on: 10-05-2015
 *      Author: Karol Dzitkowski
 */

#include "cuda_array_statistics.hpp"
#include "util/transform/cuda_array_transform.hpp"
#include "test/unittest_base.hpp"
#include "core/cuda_array.hpp"


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

	FloatingPointToIntegerOperator<float, int> opMul { precision };
	SharedCudaPtr<int> transformed = CudaArrayTransform()
		.Transform<float, int>(randomData, opMul);

	IntegerToFloatingPointOperator<int, float> opDiv { precision };
	SharedCudaPtr<float> transformedBack = CudaArrayTransform()
		.Transform<int, float>(transformed, opDiv);

	EXPECT_TRUE(
		CompareDeviceArrays(randomData->get(), transformedBack->get(), randomData->size())
			);
}

TEST_F(CudaArrayStatisticsTest, Sorted_FakeData_Int_NotSorted)
{
	int N = GetSize();
	int* h_data = new int[N];
	for(int i = 0; i < N; i++)
	{
		h_data[i] = i / 100;
		if(i % 50) h_data[i] = N;
	}
	auto d_data = CudaPtr<int>::make_shared(N);
	d_data->fillFromHost(h_data, N);

	bool expected = false;
	bool actual = CudaArrayStatistics().Sorted(d_data);

	EXPECT_EQ(expected, actual);

	delete [] h_data;
}

TEST_F(CudaArrayStatisticsTest, Sorted_FakeData_Int_Sorted)
{
	int N = GetSize();
	int* h_data = new int[N];
	for(int i = 0; i < N; i++)
		h_data[i] = i / 100 + 39;
	auto d_data = CudaPtr<int>::make_shared(N);
	d_data->fillFromHost(h_data, N);

	bool expected = true;
	bool actual = CudaArrayStatistics().Sorted(d_data);

	EXPECT_EQ(expected, actual);

	delete [] h_data;
}

TEST_F(CudaArrayStatisticsTest, RlMetric_FakeData_Int_Const)
{
	int N = GetSize();
	int* h_data = new int[N];
	for(int i = 0; i < N; i++)
		h_data[i] = 100;
	auto d_data = CudaPtr<int>::make_shared(N);
	d_data->fillFromHost(h_data, N);

	float expected = 4;
	float actual = CudaArrayStatistics().RlMetric<int, 4>(d_data);

	EXPECT_EQ(expected, actual);

	delete [] h_data;
}

TEST_F(CudaArrayStatisticsTest, RlMetric_FakeData_Int_Consecutive)
{
	float expected = 1;
	float actual = CudaArrayStatistics().RlMetric<int, 4>(GetIntConsecutiveData());
	EXPECT_EQ(expected, actual);
}

TEST_F(CudaArrayStatisticsTest, RlMetric_FakeData_Int_2ofEach)
{
	int N = GetSize();
	int* h_data = new int[N];
	for(int i = 0; i < N; i++)
		h_data[i] = i/2;
	auto d_data = CudaPtr<int>::make_shared(N);
	d_data->fillFromHost(h_data, N);

	float expected = 1.5;
	float actual = CudaArrayStatistics().RlMetric(d_data);

	EXPECT_EQ(expected, actual);

	delete [] h_data;
}

TEST_F(CudaArrayStatisticsTest, Mean_Consecutive_Int)
{
	int N = GetSize();
	float expected = (N-1) / 2;
	float actual = CudaArrayStatistics().Mean(GetIntConsecutiveData());
	EXPECT_EQ(expected, actual);
}

TEST_F(CudaArrayStatisticsTest, GenerateStatistics_noexception_Double_RandomData)
{
	CudaArrayStatistics().GenerateStatistics(
			CastSharedCudaPtr<double, char>(GetDoubleRandomData()),
			DataType::d_double);
}

} /* namespace ddj */
