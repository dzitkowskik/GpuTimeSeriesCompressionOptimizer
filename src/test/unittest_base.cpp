#include "unittest_base.hpp"

namespace ddj {

void UnittestBase::SetUpTestCase()
{
	HelperDevice hc;
	int devId = hc.SetCudaDeviceWithMaxFreeMem();
	printf("TEST SET UP ON DEVICE %d\n", devId);
}

void UnittestBase::TearDownTestCase()
{
}

void UnittestBase::SetUp()
{
	_size = 10000;
}

void UnittestBase::TearDown(){}

SharedCudaPtr<int> UnittestBase::GetIntRandomData()
{ return _generator.GenerateRandomIntDeviceArray(_size, 100, 1000); }

SharedCudaPtr<int> UnittestBase::GetIntConsecutiveData()
{ return _generator.GenerateConsecutiveIntDeviceArray(_size); }

SharedCudaPtr<float> UnittestBase::GetFloatRandomData()
{ return _generator.GenerateRandomFloatDeviceArray(_size); }

SharedCudaPtr<time_t> UnittestBase::GetTsIntDataFromTestFile()
{
	auto dataFilePath = ddj::Config::GetInstance()->GetValue<std::string>("TEST_DATA_LOG");
	File file(dataFilePath);
	TSFileDefinition fileDefinition;
	auto tsVector = TimeSeries<float>::ReadManyFromFile(file, fileDefinition);
	auto intData = tsVector[0].GetTime();
	auto size = tsVector[0].GetSize();

	auto result = CudaPtr<time_t>::make_shared();
	result->fillFromHost(intData, size);

	return result;
}

SharedCudaPtr<int> UnittestBase::GetRandomStencilData()
{
	return _generator.GenerateRandomStencil(_size);
}

SharedCudaPtr<int> UnittestBase::GetFakeIntDataForHistogram()
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

int UnittestBase::GetSize() { return _size; }

SharedCudaPtr<float> UnittestBase::GetFloatRandomDataWithMaxPrecision(int maxPrecision)
{
	return _generator.CreateRandomFloatsWithMaxPrecision(_size, maxPrecision);
}

} /* namespace ddj */
