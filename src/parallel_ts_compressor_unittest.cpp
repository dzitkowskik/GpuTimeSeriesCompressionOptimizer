/*
 * parallel_ts_compressor_unittest.cpp
 *
 *  Created on: Dec 29, 2015
 *      Author: Karol Dzitkowski
 */

#include "parallel_ts_compressor.hpp"
#include "time_series.hpp"
#include "test/unittest_base.hpp"

#include <gtest/gtest.h>

namespace ddj
{

class ParallelTsCompressorTest : public UnittestBase {};

// ./gpuStore --gtest_filter=ParallelTsCompressorTest.Compress_Info_Test_Data_no_exception
//TEST_F(ParallelTsCompressorTest, Compress_Info_Test_Data_no_exception)
//{
//	auto inputFile = File("sample_data/info.log");
//	auto outputFile = File("sample_data/info.cmpr");
//	auto headerFile = File("sample_data/info.header");
//
//	auto fileDefinition = TimeSeriesReader::ReadFileDefinition(headerFile);
//	auto reader = TimeSeriesReaderBinary::make_shared(BinaryFileDefinition(fileDefinition));
//
//	ParallelTSCompressor compressor(reader);
//	compressor.Compress(inputFile, outputFile);
//}

} /* namespace ddj */
