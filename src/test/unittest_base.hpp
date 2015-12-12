/*
 *  unittest_base.hpp
 *
 *  Created on: 01-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_UNITTEST_BASE_HPP_
#define DDJ_UNITTEST_BASE_HPP_

#include "helpers/helper_device.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "core/cuda_ptr.hpp"
#include "time_series.hpp"
#include "time_series_reader.hpp"

#include <gtest/gtest.h>

namespace ddj {

class UnittestBase : public ::testing::Test
{
public:
	static void SetUpTestCase();
	static void TearDownTestCase();

protected:
	virtual void SetUp();
	virtual void TearDown();

	SharedCudaPtr<int> GetIntRandomData(int from = 100, int to = 1000);
	SharedCudaPtr<int> GetIntConsecutiveData();
	SharedCudaPtr<float> GetFloatRandomData();
	SharedCudaPtr<double> GetDoubleRandomData();
	SharedCudaPtr<float> GetFloatRandomDataWithMaxPrecision(int maxPrecision);
	SharedCudaPtr<time_t> GetTsIntDataFromTestFile();
	SharedCudaPtr<time_t> GetNextTsIntDataFromTestFile();
	SharedCudaPtr<float> GetTsFloatDataFromTestFile();
	SharedCudaPtr<int> GetRandomStencilData();
	SharedCudaPtr<int> GetFakeIntDataForHistogram();
	int GetSize();

protected:
	CudaArrayGenerator _generator;
	TimeSeriesReader _tsReader;
	int _size;
};

} /* namespace ddj */
#endif /* DDJ_UNITTEST_BASE_HPP_ */
