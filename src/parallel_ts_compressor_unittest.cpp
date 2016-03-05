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
TEST_F(ParallelTsCompressorTest, DISABLED_Compress_Binary_Info_Test_Data_no_exception)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, Compress_Binary_Info_Test_Data_no_exception");
	auto inputFile = File("sample_data/info.inf");
	auto outputFile = File::GetTempFile();
	auto headerFile = File("sample_data/info.header");

	auto fileDefinition = TimeSeriesReader::ReadFileDefinition(headerFile);
	auto reader = TimeSeriesReaderBinary::make_shared(BinaryFileDefinition(fileDefinition), 4);

	ParallelTSCompressor compressor(reader);
	compressor.SetBatchSize(1e6);
	compressor.Compress(inputFile, outputFile);

	LOG4CPLUS_INFO_FMT(_logger, "Compression input = %s with size %lu",
		inputFile.GetPath().c_str(), inputFile.GetSize()/1024);
	LOG4CPLUS_INFO_FMT(_logger, "Compression output = %s with size %lu",
		outputFile.GetPath().c_str(), outputFile.GetSize()/1024);
	LOG4CPLUS_INFO_FMT(_logger, "Compression ratio = %f",
			(float)inputFile.GetSize()/outputFile.GetSize());
}

TEST_F(ParallelTsCompressorTest, DISABLED_Compress_CSV_Info_Test_Data_no_exception)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, Compress_CSV_Info_Test_Data_no_exception");
	auto inputFile = File("sample_data/info.log");
	auto outputFile = File::GetTempFile();
	auto headerFile = File("sample_data/info.header");

	auto fileDefinition = TimeSeriesReader::ReadFileDefinition(headerFile);
	auto reader = TimeSeriesReaderCSV::make_shared(CSVFileDefinition(fileDefinition));

	ParallelTSCompressor compressor(reader);
	compressor.SetBatchSize(1e6);
	compressor.Compress(inputFile, outputFile);

	LOG4CPLUS_INFO_FMT(_logger, "Compression input = %s with size %lu",
		inputFile.GetPath().c_str(), inputFile.GetSize()/1024);
	LOG4CPLUS_INFO_FMT(_logger, "Compression output = %s with size %lu",
		outputFile.GetPath().c_str(), outputFile.GetSize()/1024);
	LOG4CPLUS_INFO_FMT(_logger, "Compression ratio = %f",
			(float)inputFile.GetSize()/outputFile.GetSize());
}

TEST_F(ParallelTsCompressorTest, DISABLED_Decompress_Binary_Info_Test_Data_CompareFile)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, Decompress_Binary_Info_Test_Data_CompareFile");
	auto inputFile = File("sample_data/info.inf");
	auto outputFileCompr = File::GetTempFile();
	auto outputFileDecompr = File::GetTempFile();
	auto headerFile = File("sample_data/info.header");

	auto fileDefinition = BinaryFileDefinition(TimeSeriesReader::ReadFileDefinition(headerFile));
	auto reader = TimeSeriesReaderBinary::make_shared(fileDefinition, 4);
	ParallelTSCompressor compressor(reader);
	compressor.SetBatchSize(2*1e5);
	compressor.Compress(inputFile, outputFileCompr);

	LOG4CPLUS_INFO_FMT(_logger, "Compression input = %s with size %lu",
		inputFile.GetPath().c_str(), inputFile.GetSize()/1024);
	LOG4CPLUS_INFO_FMT(_logger, "Compression output = %s with size %lu",
		outputFileCompr.GetPath().c_str(), outputFileCompr.GetSize()/1024);

	compressor.Decompress(outputFileCompr, outputFileDecompr, fileDefinition);

	LOG4CPLUS_INFO_FMT(_logger, "Decompression output = %s with size %lu",
		outputFileDecompr.GetPath().c_str(), outputFileDecompr.GetSize()/1024);

	auto newInputFile = File("sample_data/info.inf");
	auto ts1 = TimeSeriesReaderBinary(fileDefinition, 4).Read(newInputFile);
	auto ts2 = TimeSeriesReaderBinary(fileDefinition, 4).Read(outputFileDecompr);

	EXPECT_TRUE( ts1->compare(*ts2) );

	LOG4CPLUS_INFO_FMT(_logger, "Compression ratio = %f",
			(float)inputFile.GetSize()/outputFileCompr.GetSize());
}

TEST_F(ParallelTsCompressorTest, DISABLED_Decompress_CSV_Info_Test_Data_CompareFile)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, Decompress_CSV_Info_Test_Data_CompareFile");
	auto inputFile = File("sample_data/info.log");
	auto outputFileCompr = File::GetTempFile();
	auto outputFileDecompr = File::GetTempFile();
	auto headerFile = File("sample_data/info.header");

	auto fileDefinition = TimeSeriesReader::ReadFileDefinition(headerFile);
	auto reader = TimeSeriesReaderCSV::make_shared(CSVFileDefinition(fileDefinition));

	ParallelTSCompressor compressor(reader);
	compressor.SetBatchSize(1e6);
	compressor.Compress(inputFile, outputFileCompr);

	LOG4CPLUS_INFO_FMT(_logger, "Compression input = %s with size %lu",
		inputFile.GetPath().c_str(), inputFile.GetSize()/1024);
	LOG4CPLUS_INFO_FMT(_logger, "Compression output = %s with size %lu",
		outputFileCompr.GetPath().c_str(), outputFileCompr.GetSize()/1024);

	compressor.Decompress(outputFileCompr, outputFileDecompr, fileDefinition);

	LOG4CPLUS_INFO_FMT(_logger, "Decompression output = %s with size %lu",
		outputFileDecompr.GetPath().c_str(), outputFileDecompr.GetSize()/1024);

	EXPECT_TRUE( inputFile.Compare(outputFileDecompr) );

	LOG4CPLUS_INFO_FMT(_logger, "Compression ratio = %f",
			(float)inputFile.GetSize()/outputFileCompr.GetSize());
}

TEST_F(ParallelTsCompressorTest, DISABLED_CompressDecompress_CSV_NYSE_CompareFile)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressDecompress_CSV_NYSE_CompareFile");
	Save1MFrom1GNyseDataInSampleData(5*1e6);

	auto inputFile = File("sample_data/nyse.csv");
	auto outputFileCompr = File::GetTempFile();
	auto outputFileDecompr = File::GetTempFile();
	auto headerFile = File("sample_data/nyse.header");

	auto fileDefinition = TimeSeriesReader::ReadFileDefinition(headerFile);
	auto reader = TimeSeriesReaderCSV::make_shared(CSVFileDefinition(fileDefinition));

	ParallelTSCompressor compressor(reader);
	compressor.SetBatchSize(1e5);
	compressor.Compress(inputFile, outputFileCompr);

	LOG4CPLUS_INFO_FMT(_logger, "Compression input = %s with size %lu",
		inputFile.GetPath().c_str(), inputFile.GetSize()/1024);
	LOG4CPLUS_INFO_FMT(_logger, "Compression output = %s with size %lu",
		outputFileCompr.GetPath().c_str(), outputFileCompr.GetSize()/1024);

	compressor.Decompress(outputFileCompr, outputFileDecompr, fileDefinition);

	LOG4CPLUS_INFO_FMT(_logger, "Decompression output = %s with size %lu",
		outputFileDecompr.GetPath().c_str(), outputFileDecompr.GetSize()/1024);

	EXPECT_TRUE( inputFile.Compare(outputFileDecompr) );

	LOG4CPLUS_INFO_FMT(_logger, "Compression ratio = %f",
			(float)inputFile.GetSize()/outputFileCompr.GetSize());
}

TEST_F(ParallelTsCompressorTest, DISABLED_CompressDecompress_Binary_NYSE_CompareFile)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressDecompress_Binary_NYSE_CompareFile");
	Save1MFrom1GNyseDataInSampleData(5*1e6);

	auto inputFile = File("sample_data/nyse.inf");
	auto outputFileCompr = File::GetTempFile();
	auto outputFileDecompr = File::GetTempFile();
	auto headerFile = File("sample_data/nyse.header");

	auto fileDefinition = BinaryFileDefinition(TimeSeriesReader::ReadFileDefinition(headerFile));
	auto reader = TimeSeriesReaderBinary::make_shared(fileDefinition);
	ParallelTSCompressor compressor(reader);
	compressor.SetBatchSize(5*1e5);
	compressor.Compress(inputFile, outputFileCompr);

	LOG4CPLUS_INFO_FMT(_logger, "Compression input = %s with size %lu",
		inputFile.GetPath().c_str(), inputFile.GetSize()/1024);
	LOG4CPLUS_INFO_FMT(_logger, "Compression output = %s with size %lu",
		outputFileCompr.GetPath().c_str(), outputFileCompr.GetSize()/1024);

	compressor.Decompress(outputFileCompr, outputFileDecompr, fileDefinition);

	LOG4CPLUS_INFO_FMT(_logger, "Decompression output = %s with size %lu",
		outputFileDecompr.GetPath().c_str(), outputFileDecompr.GetSize()/1024);

	auto newInputFile = File("sample_data/nyse.inf");
	auto ts1 = TimeSeriesReaderBinary(fileDefinition).Read(newInputFile);
	auto ts2 = TimeSeriesReaderBinary(fileDefinition).Read(outputFileDecompr);

	EXPECT_TRUE( ts1->compare(*ts2) );

	LOG4CPLUS_INFO_FMT(_logger, "Compression ratio = %f",
			(float)inputFile.GetSize()/outputFileCompr.GetSize());
}

TEST_F(ParallelTsCompressorTest, DISABLED_CompressDecompress_Binary_Generated_CompareFile)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressDecompress_Binary_Generated_CompareFile");

	auto inputFile = File("sample_data/generated.inf");
	auto outputFileCompr = File::GetTempFile();
	auto outputFileDecompr = File::GetTempFile();
	auto headerFile = File("sample_data/generated.header");

	auto fileDefinition = BinaryFileDefinition(TimeSeriesReader::ReadFileDefinition(headerFile));
	auto reader = TimeSeriesReaderBinary::make_shared(fileDefinition);
	ParallelTSCompressor compressor(reader);
	compressor.SetBatchSize(1e5);
	compressor.Compress(inputFile, outputFileCompr);

	LOG4CPLUS_INFO_FMT(_logger, "Compression input = %s with size %lu",
		inputFile.GetPath().c_str(), inputFile.GetSize()/1024);
	LOG4CPLUS_INFO_FMT(_logger, "Compression output = %s with size %lu",
		outputFileCompr.GetPath().c_str(), outputFileCompr.GetSize()/1024);

	compressor.Decompress(outputFileCompr, outputFileDecompr, fileDefinition);

	LOG4CPLUS_INFO_FMT(_logger, "Decompression output = %s with size %lu",
		outputFileDecompr.GetPath().c_str(), outputFileDecompr.GetSize()/1024);

	auto newInputFile = File("sample_data/generated.inf");
	auto ts1 = TimeSeriesReaderBinary(fileDefinition).Read(newInputFile);
	auto ts2 = TimeSeriesReaderBinary(fileDefinition).Read(outputFileDecompr);

	EXPECT_TRUE( ts1->compare(*ts2) );

	LOG4CPLUS_INFO_FMT(_logger, "Compression ratio = %f",
			(float)inputFile.GetSize()/outputFileCompr.GetSize());
}

TEST_F(ParallelTsCompressorTest, DISABLED_CompressDecompress_CSV_Generated_CompareFile)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressDecompress_CSV_Generated_CompareFile");

	auto inputFile = File("sample_data/generated.csv");
	auto outputFileCompr = File::GetTempFile();
	auto outputFileDecompr = File::GetTempFile();
	auto headerFile = File("sample_data/generated.header");

	auto fileDefinition = TimeSeriesReader::ReadFileDefinition(headerFile);
	auto reader = TimeSeriesReaderCSV::make_shared(CSVFileDefinition(fileDefinition));

	ParallelTSCompressor compressor(reader);
	compressor.SetBatchSize(1e5);
	compressor.Compress(inputFile, outputFileCompr);

	LOG4CPLUS_INFO_FMT(_logger, "Compression input = %s with size %lu",
		inputFile.GetPath().c_str(), inputFile.GetSize()/1024);
	LOG4CPLUS_INFO_FMT(_logger, "Compression output = %s with size %lu",
		outputFileCompr.GetPath().c_str(), outputFileCompr.GetSize()/1024);

	compressor.Decompress(outputFileCompr, outputFileDecompr, fileDefinition);

	LOG4CPLUS_INFO_FMT(_logger, "Decompression output = %s with size %lu",
		outputFileDecompr.GetPath().c_str(), outputFileDecompr.GetSize()/1024);

	EXPECT_TRUE( inputFile.Compare(outputFileDecompr) );

	LOG4CPLUS_INFO_FMT(_logger, "Compression ratio = %f",
			(float)inputFile.GetSize()/outputFileCompr.GetSize());
}

TEST_F(ParallelTsCompressorTest, DISABLED_CompressDecompress_CSV_browsermarket01)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressDecompress_CSV_Generated_CompareFile");

	auto inputFile = File("sample_data/browsermarket01.csv");
	auto outputFileCompr = File::GetTempFile();
	auto outputFileDecompr = File::GetTempFile();
	auto headerFile = File("sample_data/browsermarket01.header");

	auto fileDefinition = TimeSeriesReader::ReadFileDefinition(headerFile);
	auto reader = TimeSeriesReaderCSV::make_shared(CSVFileDefinition(fileDefinition));

	ParallelTSCompressor compressor(reader);
	compressor.SetBatchSize(1e5);
	compressor.Compress(inputFile, outputFileCompr);

	LOG4CPLUS_INFO_FMT(_logger, "Compression input = %s with size %lu",
		inputFile.GetPath().c_str(), inputFile.GetSize()/1024);
	LOG4CPLUS_INFO_FMT(_logger, "Compression output = %s with size %lu",
		outputFileCompr.GetPath().c_str(), outputFileCompr.GetSize()/1024);

	compressor.Decompress(outputFileCompr, outputFileDecompr, fileDefinition);

	LOG4CPLUS_INFO_FMT(_logger, "Decompression output = %s with size %lu",
		outputFileDecompr.GetPath().c_str(), outputFileDecompr.GetSize()/1024);

	LOG4CPLUS_INFO_FMT(_logger, "Compression ratio = %f",
			(float)inputFile.GetSize()/outputFileCompr.GetSize());
}

} /* namespace ddj */
