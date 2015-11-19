/*
 * benchmark_base.cpp
 *
 *  Created on: 11 lis 2015
 *      Author: ghash
 */

#include "benchmarks/benchmark_base.hpp"
#include "core/config.hpp"

namespace ddj
{

SharedCudaPtr<int> BenchmarkBase::GetIntRandomData(int n, int from, int to)
{ return _generator.GenerateRandomIntDeviceArray(n, from, to); }

SharedCudaPtr<int> BenchmarkBase::GetIntConsecutiveData(int n)
{ return _generator.GenerateConsecutiveIntDeviceArray(n); }

SharedCudaPtr<float> BenchmarkBase::GetFloatRandomData(int n)
{ return _generator.GenerateRandomFloatDeviceArray(n); }

SharedCudaPtr<time_t> BenchmarkBase::GetTsIntDataFromFile(int n)
{
	auto dataFilePath = ddj::Config::GetInstance()->GetValue<std::string>("BENCHMARK_DATA_LOG");
	File file(dataFilePath);
	CSVFileDefinition fileDefinition;
	auto tsVector = TimeSeries<float>::ReadFromCSV(file, fileDefinition, n);
	auto intData = tsVector[0].GetTimeRaw();
	auto size = tsVector[0].GetSize();

	auto result = CudaPtr<time_t>::make_shared();
	result->fillFromHost(intData, size);

	return result;
}

SharedCudaPtr<int> BenchmarkBase::GetFakeIntDataForHistogram(int n)
{
	int mod = n / 10;
	int big = n / 3;
	std::vector<int> h_fakeData;
	for(int i = 0; i < n; i++)
	{
		if(i%mod == 0 || i%mod == 1)
			h_fakeData.push_back(n);
		else
			h_fakeData.push_back(i%mod);
	}
	auto fakeData = CudaPtr<int>::make_shared(n);
	fakeData->fillFromHost(h_fakeData.data(), n);
	return fakeData;
}

SharedCudaPtr<float> BenchmarkBase::GetTsFloatDataFromFile(int n)
{
	auto dataFilePath = ddj::Config::GetInstance()->GetValue<std::string>("BENCHMARK_DATA_LOG");
	File file(dataFilePath);
	CSVFileDefinition fileDefinition;
	auto tsVector = TimeSeries<float>::ReadFromCSV(file, fileDefinition, n);
	auto floatData = tsVector[0].GetDataRaw();
	auto size = tsVector[0].GetSize();

	auto result = CudaPtr<float>::make_shared();
	result->fillFromHost(floatData, size);

	return result;
}

SharedCudaPtr<float> BenchmarkBase::GetFloatRandomDataWithMaxPrecision(int n, int maxPrecision)
{
	return _generator.CreateRandomFloatsWithMaxPrecision(n, maxPrecision);
}

void BenchmarkBase::SetStatistics(benchmark::State& state, DataType type)
{
	auto it_processed = static_cast<int64_t>(state.iterations() * state.range_x());
	state.SetItemsProcessed(it_processed);
	state.SetBytesProcessed(it_processed * GetDataTypeSize(type));
}


} /* namespace ddj */
