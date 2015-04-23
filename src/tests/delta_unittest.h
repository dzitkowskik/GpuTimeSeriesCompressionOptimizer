/*
 * delta_unittest.h
 *
 *  Created on: 22-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_DELTA_UNITTEST_H_
#define DDJ_DELTA_UNITTEST_H_

#include "compression/delta/delta_encoding.h"
#include "helpers/helper_cuda.h"
#include "helpers/helper_macros.h"
#include "helpers/helper_generator.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>

namespace ddj
{

class DeltaEncodingTest : public testing::Test,
    public ::testing::WithParamInterface<int>
{
public:
	DeltaEncodingTest()
	{
		HelperCuda hc;
		hc.SetCudaDeviceWithMaxFreeMem();
		d_random_data = NULL;
	}
	virtual ~DeltaEncodingTest(){}

	virtual void SetUp()
	{
		int n = GetParam();
		curandGenerator_t gen;
		d_random_data = generator.GenerateRandomFloatDeviceArray(n);
	}

	virtual void TearDown()
	{
		CUDA_CALL(cudaFree(d_random_data));
	}

	float* d_random_data;
	DeltaEncoding compression;

private:
	HelperGenerator generator;
};

} /* namespace ddj */
#endif /* DDJ_DELTA_UNITTEST_H_ */
