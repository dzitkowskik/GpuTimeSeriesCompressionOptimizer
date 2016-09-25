/*
 *  benchmark_base.hpp
 *
 *  Created on: 11/11/2015
 *      Author: Karol Dzitkowski
 */

#ifndef BENCHMARK_BASE_HPP_
#define BENCHMARK_BASE_HPP_

#include "core/cuda_ptr.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "data/time_series_reader.hpp"
#include "data/data_type.hpp"

#include <benchmark/benchmark.h>

namespace ddj
{

class BenchmarkBase : public ::benchmark::Fixture
{
public:
	BenchmarkBase()
	{
		CSVFileDefinition csvFileDefinition;
		csvFileDefinition.Columns = std::vector<DataType> {
	            DataType::d_time,
	            DataType::d_float,
	            DataType::d_float,
	            DataType::d_float
	    };

		_tsReaderCSV = TimeSeriesReaderCSV(csvFileDefinition);
	}
	~BenchmarkBase(){}

protected:
	SharedCudaPtr<int> GetIntRandomData(int n, int from = 100, int to = 1000);
	SharedCudaPtr<int> GetIntConsecutiveData(int n);
	SharedCudaPtr<float> GetFloatRandomData(int n);
	SharedCudaPtr<float> GetFloatRandomDataWithMaxPrecision(int n, int maxPrecision);
	SharedCudaPtr<time_t> GetTsIntDataFromFile(int n);
	SharedCudaPtr<float> GetTsFloatDataFromFile(int n);
	SharedCudaPtr<int> GetFakeIntDataForHistogram(int n);
	SharedCudaPtr<time_t> GetFakeDataForTime(size_t n)
	{
		return _generator.GetFakeDataForTime(1e5, 0.1f, n);
	}

	template<typename T>
	SharedCudaPtr<T> GetFakeDataWithPatternA(int p, size_t n)
	{
		return _generator.GetFakeDataWithPatternA(p, (size_t)1e2, (T)5, (T)1, (T)1e6, n);
	}

	template<typename T>
	SharedCudaPtr<T> GetFakeDataWithPatternB(int p, size_t n)
	{
		return _generator.GetFakeDataWithPatternB(p, n, (T)3, (T)1e7, n);
	}

	void SetStatistics(benchmark::State& state, DataType type);

protected:
	TimeSeriesReaderCSV _tsReaderCSV;
	CudaArrayGenerator _generator;
};

} /* namespace ddj */
#endif /* BENCHMARK_BASE_HPP_ */
