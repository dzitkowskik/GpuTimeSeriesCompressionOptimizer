#include "unittest_base.hpp"
#include "core/macros.h"

namespace ddj {

void UnittestBase::SetUpTestCase()
{
	CudaDevice hc;
	int devId = hc.SetCudaDeviceWithMaxFreeMem();
	auto logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("UnittestBase"));
	LOG4CPLUS_INFO_FMT(logger, "TEST SET UP ON DEVICE %d\n", devId);
	cudaGetLastError();
}

void UnittestBase::TearDownTestCase()
{
	cudaGetLastError();
}

void UnittestBase::SetUp()
{
	CUDA_ASSERT_RETURN( cudaGetLastError() );
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

bool fileExists(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

void UnittestBase::Save1MFrom1GNyseDataInSampleData(size_t size)
{
	if(size == 0) size = _size;

	BinaryFileDefinition fileDefinition;
	fileDefinition.Columns = this->_nyseData;
	fileDefinition.Header = this->_nyseDataHeader;

	auto dataFilePath = ddj::Config::GetInstance()->GetValue<std::string>("NYSE_DATA_1GB");
	File file(dataFilePath);

	File outputBin("sample_data/nyse.inf");
	File outputCsv("sample_data/nyse.csv");
	File outputDef("sample_data/nyse.header");

	if(fileExists(outputCsv.GetPath().c_str())) return;

	auto ts = TimeSeriesReaderBinary(fileDefinition).Read(file, size);
	TimeSeriesReaderBinary(fileDefinition).Write(outputBin, *ts);
	TimeSeriesReaderCSV(fileDefinition).Write(outputCsv, *ts);

	TimeSeriesReader::WriteFileDefinition(outputDef, fileDefinition);
}

} /* namespace ddj */
