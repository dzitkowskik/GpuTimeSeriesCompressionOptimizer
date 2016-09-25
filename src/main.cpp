/*
 * main.cpp
 *
 * Created on: Mar 10, 2015
 * Author: Karol Dzitkowski
 */

#include "parallel_ts_compressor.hpp"
#include "data/time_series_reader.hpp"
#include "core/logger.h"
#include "core/config.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include <signal.h>
#include <boost/bind.hpp>

#define DEFAULT_GENERATE_SIZE 1e6

using namespace std;
using namespace ddj;
namespace po = boost::program_options;

ConfigOptions GetProgramOptions()
{
	ConfigOptions options;

	// SET CONSOLE OPTIONS
	po::options_description consoleOptions("Config file options");
	consoleOptions.add_options()
			("compress,c", "compress")
			("decompress,d", "decompress")
			("header-file,h", po::value<string>(), "header file")
			("input-file,i", po::value<string>(), "input file")
			("output-file,o", po::value<string>(), "output file")
			("generate,g", po::value<int>()->default_value(DEFAULT_GENERATE_SIZE), "generate sample data")
			("padding,p", po::value<int>(), "binary data padding")
			("format,f", po::value<string>()->default_value("bin"), "input format")
			;
	options.ConsoleOptions.add(consoleOptions);

	// SET CONSOLE POSITIONAL OPTIONS
	po::positional_options_description consolePositionalOptions;
	consolePositionalOptions.add("input-file", 1);
	consolePositionalOptions.add("output-file", 2);
	options.ConsolePositionalOptions = consolePositionalOptions;

	// SET CONFIG FILE OPTIONS
	po::options_description configFileOptions("Config file options");
	options.ConfigFileOptions.add(configFileOptions);

	return options;
}

void initialize_logger()
{
	log4cplus::initialize();
	LogLog::getLogLog()->setInternalDebugging(true);
	PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT("logger.prop"));
}

void generate(size_t size);

boost::shared_ptr<TimeSeriesReader> getReader(FileDefinition& fileDefinition)
{
	boost::shared_ptr<TimeSeriesReader> result;
	auto conf = ddj::Config::GetInstance();
	bool useBinaryFormat = conf->GetValue<std::string>("format").compare("bin") == 0;

	// get padding if available
	if(useBinaryFormat)
	{
		int padding = 0;
		if(conf->HasValue("padding")) padding = conf->GetValue<int>("padding");
		result = TimeSeriesReaderBinary::make_shared(BinaryFileDefinition(fileDefinition), padding);
	}
	else result = TimeSeriesReaderCSV::make_shared(CSVFileDefinition(fileDefinition));

	return result;
}

int main(int argc, char* argv[])
{
	// Configure program
	ConfigDefinition configDef { argc, argv, "", GetProgramOptions() };
	ddj::Config::Initialize(configDef);

	initialize_logger();

	auto conf = ddj::Config::GetInstance();

	auto inputFile = File(conf->GetValue<std::string>("input-file"));
	auto outputFile = File(conf->GetValue<std::string>("output-file"));
	auto headerFile = File(conf->GetValue<std::string>("header-file"));
	auto fileDefinition = TimeSeriesReader::ReadFileDefinition(headerFile);

	printf("Input file: %s\n", inputFile.GetPath().c_str());
	printf("Output file: %s\n", outputFile.GetPath().c_str());
	printf("Header file: %s\n", headerFile.GetPath().c_str());

	ParallelTSCompressor compressor(getReader(fileDefinition));

	if(conf->HasValue("compress")) // Compress
	{
		printf("START COMPRESSING\n");
		compressor.Compress(inputFile, outputFile);
		printf("COMPRESSING DONE\n");
	}
	else if (conf->HasValue("decompress"))
	{
		printf("START DECOMPRESSING\n");
		compressor.Decompress(inputFile, outputFile, fileDefinition);
		printf("DECOMPRESSING DONE\n");
	}
	else if (conf->HasValue("generate"))
	{
		printf("START GENERATING DATA\n");
		size_t size = conf->GetValue<int>("generate");
		generate(size);
		printf("GENERATING %lu DATA DONE\n", size);
	}
	else std::cout << configDef.Options.ConsoleOptions << std::endl;


	return 0;
}

void generate(size_t size)
{
	BinaryFileDefinition def;
	def.Header = std::vector<std::string> {
	    	"TIME",
	    	"PATTERN_A_INT",
	    	"PATTERN_B_INT",
	    	"FLOAT_PREC_4",
	    	"PATTERN_A_FLOAT",
	    	"PATTERN_B_FLOAT"
	};
	def.Columns = std::vector<DataType> {
	    	DataType::d_time,
	    	DataType::d_int,
	    	DataType::d_int,
	    	DataType::d_float,
	    	DataType::d_float,
	    	DataType::d_float
	};
	def.Decimals = std::vector<int> {0, 0, 0, 4, 6, 6};

	CudaArrayGenerator gen;
	auto c0 = CastSharedCudaPtr<time_t, char>(
			gen.GetFakeDataForTime(time(NULL), 0.05, size));
	auto c1 = CastSharedCudaPtr<int, char>(
			gen.GetFakeDataWithPatternA(0, 10, 10, 0, 1000000, size));
	auto c2 = CastSharedCudaPtr<int, char>(
			gen.GetFakeDataWithPatternB(0, 100000, 0, 1000000, size));
	auto c3 = CastSharedCudaPtr<float, char>(
			gen.CreateRandomFloatsWithMaxPrecision(size, 4));
	auto c4 = CastSharedCudaPtr<float, char>(
			gen.GetFakeDataWithPatternA<float>(0, 10, 0.01, 0., 1e3, size));
	auto c5 = CastSharedCudaPtr<float, char>(
			gen.GetFakeDataWithPatternB<float>(0, 1e6, 0, 1e7, size));
	auto data = SharedCudaPtrVector<char> { c0, c1, c2, c3, c4, c5 };
	SharedTimeSeriesPtr ts = TimeSeries::make_shared(def);
	for(int i = 0; i < data.size(); i++)
	{
		ts->getColumn(i).reserveSize(data[i]->size());
		char* h_column = ts->getColumn(i).getData();
		CUDA_CALL( cudaMemcpy(h_column, data[i]->get(), data[i]->size(), CPY_DTH) );
	}
	ts->updateRecordsCnt();
	// CREATE CSV
	auto reader = TimeSeriesReaderCSV::make_shared(CSVFileDefinition(def));
	auto outputFile = File("sample_data/generated.csv");
	reader->Write(outputFile, *ts);

	// CREATE BINARY
	auto reader2 = TimeSeriesReaderBinary::make_shared(BinaryFileDefinition(def));
	auto outputFile2 = File("sample_data/generated.inf");
	reader2->Write(outputFile2, *ts);
}
