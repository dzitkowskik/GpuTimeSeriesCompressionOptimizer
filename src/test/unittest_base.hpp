/*
 *  unittest_base.hpp
 *
 *  Created on: 01-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_UNITTEST_BASE_HPP_
#define DDJ_UNITTEST_BASE_HPP_

#include "data_type.hpp"
#include "helpers/helper_device.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "core/cuda_ptr.hpp"
#include "time_series.hpp"
#include "time_series_reader.hpp"

#include <vector>
#include <gtest/gtest.h>
#include <boost/shared_ptr.hpp>

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

	boost::shared_ptr<TimeSeries> Get1GBNyseTimeSeries();
	int GetSize();

protected:
	CudaArrayGenerator _generator;
	TimeSeriesReader _tsReader;
	int _size;

private:
	std::vector<DataType> _nyseData = std::vector<DataType> {
		DataType::d_int,   // MsgSeqNum
		DataType::d_short, // MsgType
		DataType::d_int,   // SendTime
		DataType::d_char, DataType::d_char, DataType::d_char, DataType::d_char, //Symbol - 11 characters
		DataType::d_char, DataType::d_char, DataType::d_char, DataType::d_char,
		DataType::d_char, DataType::d_char, DataType::d_char,
		DataType::d_short,  // MsgSize
		DataType::d_short,  // Security index
		DataType::d_int,  	// SourceTime
		DataType::d_short,  // SourceTimeMocroSecs
		DataType::d_char,  	// QuoteCondition
		DataType::d_char,   // TradingStatus
		DataType::d_int,  	// SourceSeqNum
		DataType::d_char,  	// SourceSessionID
		DataType::d_char,  	// PriceScaleCode
		DataType::d_int,  	// PriceNumerator
		DataType::d_int,  	// Volume
		DataType::d_int,  	// ChgQty
		DataType::d_short,	// NumOrders
		DataType::d_char,  	// Side
		DataType::d_char,  	// Filler
		DataType::d_char,  	// ReasonCode
		DataType::d_char,  	// Filler
		DataType::d_int,  	// LinkID1
		DataType::d_int,  	// LinkID2
		DataType::d_int,  	// LinkID3
	};

	std::vector<std::string> _nyseDataHeader = std::vector<std::string> {
		"MsgSeqNum",
		"MsgType",
		"SendTime",
		"Symbol 1", "Symbol 2", "Symbol 3", "Symbol 4","Symbol 5", "Symbol 6",
		"Symbol 7", "Symbol 8", "Symbol 9", "Symbol 10", "Symbol 11",
		"MsgSize",
		"Security index",
		"SourceTime",
		"SourceTimeMocroSecs",
		"QuoteCondition",
		"TradingStatus",
		"SourceSeqNum",
		"SourceSessionID",
		"PriceScaleCode",
		"PriceNumerator",
		"Volume",
		"ChgQty",
		"NumOrders",
		"Side",
		"Filler",
		"ReasonCode",
		"Filler",
		"LinkID1",
		"LinkID2",
		"LinkID3"
	};
};

} /* namespace ddj */
#endif /* DDJ_UNITTEST_BASE_HPP_ */
