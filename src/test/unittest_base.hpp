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
	static void SetUpTestCase()
	{
//		printf("SetUpTestCase()\n");
		HelperDevice hc;
        hc.SetCudaDeviceWithMaxFreeMem();
	}

protected:

	virtual void SetUp()
	{
//		printf("SetUp\n");
		_size = 100;
	}

	virtual void TearDown(){}

	SharedCudaPtr<int> GetIntRandomData()
	{ return _generator.GenerateRandomIntDeviceArray(_size, 100, 1000); }

	SharedCudaPtr<int> GetIntConsecutiveData()
	{ return _generator.GenerateConsecutiveIntDeviceArray(_size); }

	SharedCudaPtr<float> GetFloatRandomData()
	{ return _generator.GenerateRandomFloatDeviceArray(_size); }

	SharedCudaPtr<int> GetTsIntDataFromTestFile()
	{
		auto dataFilePath = ddj::Config::GetInstance()->GetValue<std::string>("TEST_DATA_LOG");
		File file(dataFilePath);
		TSFileDefinition fileDefinition;
		auto tsVector = TimeSeries<float, int>::ReadManyFromFile(file, fileDefinition);
		auto intData = tsVector[0].GetTime();
		auto size = tsVector[0].GetSize();

		auto result = CudaPtr<int>::make_shared();
		result->fillFromHost(intData, size);

		return result;
	}

	SharedCudaPtr<int> GetFakeIntDataForHistogram()
	{
	    int mod = _size / 10;
	    int big = _size/3;
	    std::vector<int> h_fakeData;
	    for(int i = 0; i < _size; i++)
	    {
	        if(i%mod == 0 || i%mod == 1)
	            h_fakeData.push_back(_size);
	        else
	            h_fakeData.push_back(i%mod);
	    }
	    auto fakeData = CudaPtr<int>::make_shared(_size);
	    fakeData->fillFromHost(h_fakeData.data(), _size);
	    return fakeData;
	}

	int GetSize() { return _size; }

	CudaArrayGenerator _generator;
	int _size;
};

} /* namespace ddj */
#endif /* DDJ_UNITTEST_BASE_HPP_ */
