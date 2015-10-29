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
#include "time_series.h"

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

	SharedCudaPtr<int> GetIntRandomData();
	SharedCudaPtr<int> GetIntConsecutiveData();
	SharedCudaPtr<float> GetFloatRandomData();
	SharedCudaPtr<float> GetFloatRandomDataWithMaxPrecision(int maxPrecision);
	SharedCudaPtr<time_t> GetTsIntDataFromTestFile();
	SharedCudaPtr<int> GetRandomStencilData();
	SharedCudaPtr<int> GetFakeIntDataForHistogram();
	int GetSize();

protected:
	CudaArrayGenerator _generator;
	int _size;
};

} /* namespace ddj */
#endif /* DDJ_UNITTEST_BASE_HPP_ */
