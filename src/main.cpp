/*
 * main.cpp
 *
 * Created on: Mar 10, 2015
 * Author: Karol Dzitkowski
 */

#include "parallel_ts_compressor.hpp"
#include "time_series_reader.hpp"
#include "core/logger.h"
#include "core/config.hpp"
#include <signal.h>
#include <boost/bind.hpp>

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

	printf("Input file: %s\n", inputFile.GetPath().c_str());
	printf("Output file: %s\n", outputFile.GetPath().c_str());
	printf("Header file: %s\n", headerFile.GetPath().c_str());

	auto fileDefinition = TimeSeriesReader::ReadFileDefinition(headerFile);
	auto reader = TimeSeriesReaderBinary::make_shared(BinaryFileDefinition(fileDefinition));

	ParallelTSCompressor compressor(reader);
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
	else std::cout << configDef.Options.ConsoleOptions << std::endl;


	return 0;
}
