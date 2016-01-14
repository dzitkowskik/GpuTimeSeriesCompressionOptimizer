#include "unittest_base.hpp"
#include "core/macros.h"

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

template<typename T>
SharedCudaPtr<T> UnittestBase::GetFakeDataWithPatternA(
		int part,
		size_t len,
		T step,
		T min,
		T max,
		size_t s)
{

	size_t size = s == 0 ? _size : s;
	auto h_result = new T[size];
	size_t start = part*size;

	// Prepare data
	min = min + (T)(start/len)*step;
	for(size_t i = 0; i < size; i++)
	{
		auto value = min;
		if((start+i) % len == 0)
		{
			value = max;
			min += step;
		}
		h_result[i] = value;
	}

	auto d_result = CudaPtr<T>::make_shared(size);
	d_result->fillFromHost(h_result, size);
	delete [] h_result;
	return d_result;
}

template<typename T>
SharedCudaPtr<T> UnittestBase::GetFakeDataWithPatternB(
		int part,
		size_t len,
		T min,
		T max,
		size_t s)
{
	int maxRand = 5;
	srand(time(NULL));
	size_t size = s == 0 ? _size : s;
	auto h_result = new T[size];
	size_t start = part*size;
	auto value = max;
	auto step = (max-min)/len/2;

	// Prepare data
	for(size_t i = 0; i < size; i++)
	{
		if((start+i)/len % 2 == 0)	// pattern1
		{
			auto randomInt = rand() % (maxRand+1);
			switch(i%2)
			{
				case 0: value = max-randomInt; break;
				case 1: value = min+randomInt; break;
			}
		} else { // pattern2
			auto x = (start+i) % len;
			value = x < (len/2) ? max - x*step : max-(len-x)*step;
		}
		h_result[i] = value;
	}

	auto d_result = CudaPtr<T>::make_shared(size);
	d_result->fillFromHost(h_result, size);
	delete [] h_result;
	return d_result;
}

SharedCudaPtr<time_t> UnittestBase::GetFakeDataForTime(
		time_t min,
		double flatness,
		size_t s)
{
	int maxStep=2;
	srand(time(NULL));
	size_t size = s == 0 ? _size : s;
	auto h_result = new time_t[size];
	auto value = min;

	// Prepare data
	for(size_t i = 0; i < size; i++)
	{
		auto step = 1 + (rand() % maxStep);
		if(rand()%100 > flatness*100) value += step;
		h_result[i] = value;
	}

	auto d_result = CudaPtr<time_t>::make_shared(size);
	d_result->fillFromHost(h_result, size);
	delete [] h_result;
	return d_result;
}

#define UNITTEST_BASE_SPEC(X) \
	template SharedCudaPtr<X> UnittestBase::GetFakeDataWithPatternA<X>(int, size_t, X, X, X, size_t); \
	template SharedCudaPtr<X> UnittestBase::GetFakeDataWithPatternB<X>(int, size_t, X, X, size_t);
FOR_EACH(UNITTEST_BASE_SPEC, short, float, int, long, long long, unsigned int)

} /* namespace ddj */
