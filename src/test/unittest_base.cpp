#include "unittest_base.hpp"

namespace ddj {

void UnittestBase::SetUpTestCase()
{
	HelperDevice hc;
	int devId = hc.SetCudaDeviceWithMaxFreeMem();
	printf("TEST SET UP ON DEVICE %d\n", devId);
	cudaGetLastError();
}

void UnittestBase::TearDownTestCase()
{
	cudaGetLastError();
}

void UnittestBase::SetUp()
{
	_size = 10000;
}

void UnittestBase::TearDown(){}

SharedCudaPtr<int> UnittestBase::GetIntRandomData(int from, int to)
{ return _generator.GenerateRandomIntDeviceArray(_size, from, to); }

SharedCudaPtr<int> UnittestBase::GetIntConsecutiveData()
{ return _generator.GenerateConsecutiveIntDeviceArray(_size); }

SharedCudaPtr<float> UnittestBase::GetFloatRandomData()
{ return _generator.GenerateRandomFloatDeviceArray(_size); }

SharedCudaPtr<double> UnittestBase::GetDoubleRandomData()
{ return _generator.GenerateRandomDoubleDeviceArray(_size); }

SharedCudaPtr<time_t> UnittestBase::GetNextTsIntDataFromTestFile()
{
	auto dataFilePath = ddj::Config::GetInstance()->GetValue<std::string>("TEST_DATA_LOG");
	File file(dataFilePath);

    auto ts = _tsReaderCSV.Read(file, _size);

	auto data = reinterpret_cast<time_t*>(ts->getColumn(0).getData());
	auto size = ts->getColumn(0).getSize() / sizeof(time_t);

	auto result = CudaPtr<time_t>::make_shared();
	result->fillFromHost(data, size);

	ts.reset();
	return result;
}

SharedCudaPtr<time_t> UnittestBase::GetTsIntDataFromTestFile()
{
	auto dataFilePath = ddj::Config::GetInstance()->GetValue<std::string>("TEST_DATA_LOG");
	File file(dataFilePath);
	CSVFileDefinition fileDefinition;
    fileDefinition.Columns = std::vector<DataType> {
            DataType::d_time,
            DataType::d_float,
            DataType::d_float,
            DataType::d_float
    };

	auto ts = TimeSeriesReaderCSV(fileDefinition).Read(file, _size);
	auto data = reinterpret_cast<time_t*>(ts->getColumn(0).getData());
	auto size = ts->getColumn(0).getSize() / sizeof(time_t);

	auto result = CudaPtr<time_t>::make_shared();
	result->fillFromHost(data, size);

	ts.reset();
	return result;
}

SharedCudaPtr<int> UnittestBase::GetRandomStencilData()
{
	return _generator.GenerateRandomStencil(_size);
}

SharedCudaPtr<int> UnittestBase::GetFakeIntDataForHistogram()
{
	int mod = _size / 10;
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

SharedCudaPtr<float> UnittestBase::GetTsFloatDataFromTestFile()
{
	auto dataFilePath = ddj::Config::GetInstance()->GetValue<std::string>("TEST_DATA_LOG");
	File file(dataFilePath);
	CSVFileDefinition fileDefinition;
    fileDefinition.Columns = std::vector<DataType> {
            DataType::d_time,
            DataType::d_float,
            DataType::d_float,
            DataType::d_float
    };

    auto ts = TimeSeriesReaderCSV(fileDefinition).Read(file, _size);

	auto data = reinterpret_cast<float*>(ts->getColumn(1).getData());
	auto size = ts->getColumn(1).getSize() / sizeof(float);

	auto result = CudaPtr<float>::make_shared();
	result->fillFromHost(data, size);

	ts.reset();
	return result;
}

int UnittestBase::GetSize() { return _size; }

SharedCudaPtr<float> UnittestBase::GetFloatRandomDataWithMaxPrecision(int maxPrecision)
{
	return _generator.CreateRandomFloatsWithMaxPrecision(_size, maxPrecision);
}

boost::shared_ptr<TimeSeries> UnittestBase::Get1GBNyseTimeSeries()
{
	auto result = boost::make_shared<TimeSeries>("NYSE");

	BinaryFileDefinition fileDefinition;
	fileDefinition.Columns = this->_nyseData;
	fileDefinition.Header = this->_nyseDataHeader;

	auto dataFilePath = ddj::Config::GetInstance()->GetValue<std::string>("NYSE_DATA_1GB");
	File file(dataFilePath);

	auto ts = TimeSeriesReaderBinary(fileDefinition).Read(file, _size);

	return ts;
}

} /* namespace ddj */
