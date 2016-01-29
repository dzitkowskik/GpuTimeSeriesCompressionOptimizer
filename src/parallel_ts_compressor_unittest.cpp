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
TEST_F(ParallelTsCompressorTest, Compress_Binary_Info_Test_Data_no_exception)
{
	auto inputFile = File("sample_data/info.inf");
	auto outputFile = File::GetTempFile();
	auto headerFile = File("sample_data/info.header");

	auto fileDefinition = TimeSeriesReader::ReadFileDefinition(headerFile);
	auto reader = TimeSeriesReaderBinary::make_shared(BinaryFileDefinition(fileDefinition), 4);

	ParallelTSCompressor compressor(reader);
	compressor.Compress(inputFile, outputFile);

	printf("Compression input = %s with size %lu\n", inputFile.GetPath().c_str(), inputFile.GetSize()/1024);
	printf("Compression output = %s with size %lu\n", outputFile.GetPath().c_str(), outputFile.GetSize()/1024);
}

TEST_F(ParallelTsCompressorTest, Compress_CSV_Info_Test_Data_no_exception)
{
	auto inputFile = File("sample_data/info.log");
	auto outputFile = File::GetTempFile();
	auto headerFile = File("sample_data/info.header");

	auto fileDefinition = TimeSeriesReader::ReadFileDefinition(headerFile);
	auto reader = TimeSeriesReaderCSV::make_shared(CSVFileDefinition(fileDefinition));

	ParallelTSCompressor compressor(reader);
	compressor.Compress(inputFile, outputFile);

	printf("Compression input = %s with size %lu\n", inputFile.GetPath().c_str(), inputFile.GetSize()/1024);
	printf("Compression output = %s with size %lu\n", outputFile.GetPath().c_str(), outputFile.GetSize()/1024);
}

TEST_F(ParallelTsCompressorTest, Decompress_Binary_Info_Test_Data_CompareFile)
{
	auto inputFile = File("sample_data/info.inf");
	auto outputFileCompr = File::GetTempFile();
	auto outputFileDecompr = File::GetTempFile();
	auto headerFile = File("sample_data/info.header");

	auto fileDefinition = BinaryFileDefinition(TimeSeriesReader::ReadFileDefinition(headerFile));
	auto reader = TimeSeriesReaderBinary::make_shared(fileDefinition, 4);
	ParallelTSCompressor compressor(reader);
	compressor.Compress(inputFile, outputFileCompr);

	printf("Compression input = %s with size %lu\n", inputFile.GetPath().c_str(), inputFile.GetSize()/1024);
	printf("Compression output = %s with size %lu\n", outputFileCompr.GetPath().c_str(), outputFileCompr.GetSize()/1024);

	compressor.Decompress(outputFileCompr, outputFileDecompr, fileDefinition);

	printf("Decompression output = %s with size %lu\n", outputFileDecompr.GetPath().c_str(), outputFileDecompr.GetSize()/1024);

	auto newInputFile = File("sample_data/info.inf");
	auto ts1 = TimeSeriesReaderBinary(fileDefinition, 4).Read(newInputFile);
	auto ts2 = TimeSeriesReaderBinary(fileDefinition, 4).Read(outputFileDecompr);

	EXPECT_TRUE( ts1->compare(*ts2) );
}

TEST_F(ParallelTsCompressorTest, Decompress_CSV_Info_Test_Data_CompareFile)
{
	auto inputFile = File("sample_data/info.log");
	auto outputFileCompr = File::GetTempFile();
	auto outputFileDecompr = File::GetTempFile();
	auto headerFile = File("sample_data/info.header");

	auto fileDefinition = TimeSeriesReader::ReadFileDefinition(headerFile);
	auto reader = TimeSeriesReaderCSV::make_shared(CSVFileDefinition(fileDefinition));

	ParallelTSCompressor compressor(reader);
	compressor.Compress(inputFile, outputFileCompr);

	printf("Compression input = %s with size %lu\n", inputFile.GetPath().c_str(), inputFile.GetSize()/1024);
	printf("Compression output = %s with size %lu\n", outputFileCompr.GetPath().c_str(), outputFileCompr.GetSize()/1024);

	compressor.Decompress(outputFileCompr, outputFileDecompr, fileDefinition);

	printf("Decompression output = %s with size %lu\n", outputFileDecompr.GetPath().c_str(), outputFileDecompr.GetSize()/1024);

	EXPECT_TRUE( inputFile.Compare(outputFileDecompr) );
}

TEST_F(ParallelTsCompressorTest, CompressDecompress_CSV_NYSE_CompareFile)
{
	Save1MFrom1GNyseDataInSampleData(1e5);

	auto inputFile = File("sample_data/nyse.csv");
	auto outputFileCompr = File::GetTempFile();
	auto outputFileDecompr = File::GetTempFile();
	auto headerFile = File("sample_data/nyse.header");

	auto fileDefinition = TimeSeriesReader::ReadFileDefinition(headerFile);
	auto reader = TimeSeriesReaderCSV::make_shared(CSVFileDefinition(fileDefinition));

	ParallelTSCompressor compressor(reader);
	compressor.Compress(inputFile, outputFileCompr);

	printf("Compression input = %s with size %lu\n", inputFile.GetPath().c_str(), inputFile.GetSize()/1024);
	printf("Compression output = %s with size %lu\n", outputFileCompr.GetPath().c_str(), outputFileCompr.GetSize()/1024);

	compressor.Decompress(outputFileCompr, outputFileDecompr, fileDefinition);

	printf("Decompression output = %s with size %lu\n", outputFileDecompr.GetPath().c_str(), outputFileDecompr.GetSize()/1024);

	EXPECT_TRUE( inputFile.Compare(outputFileDecompr) );
}

TEST_F(ParallelTsCompressorTest, CompressDecompress_Binary_NYSE_CompareFile)
{
	Save1MFrom1GNyseDataInSampleData(1e5);

	auto inputFile = File("sample_data/nyse.inf");
	auto outputFileCompr = File::GetTempFile();
	auto outputFileDecompr = File::GetTempFile();
	auto headerFile = File("sample_data/nyse.header");

	auto fileDefinition = BinaryFileDefinition(TimeSeriesReader::ReadFileDefinition(headerFile));
	auto reader = TimeSeriesReaderBinary::make_shared(fileDefinition);
	ParallelTSCompressor compressor(reader);
	compressor.Compress(inputFile, outputFileCompr);

	printf("Compression input = %s with size %lu\n", inputFile.GetPath().c_str(), inputFile.GetSize()/1024);
	printf("Compression output = %s with size %lu\n", outputFileCompr.GetPath().c_str(), outputFileCompr.GetSize()/1024);

	compressor.Decompress(outputFileCompr, outputFileDecompr, fileDefinition);

	printf("Decompression output = %s with size %lu\n", outputFileDecompr.GetPath().c_str(), outputFileDecompr.GetSize()/1024);

	auto newInputFile = File("sample_data/nyse.inf");
	auto ts1 = TimeSeriesReaderBinary(fileDefinition).Read(newInputFile);
	//	ts1->print(5);
	auto ts2 = TimeSeriesReaderBinary(fileDefinition).Read(outputFileDecompr);
	//	ts2->print(5);

	EXPECT_TRUE( ts1->compare(*ts2) );
}

} /* namespace ddj */
