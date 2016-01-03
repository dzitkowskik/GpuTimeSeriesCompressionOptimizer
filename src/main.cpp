/*
 * main.cpp
 *
 * Created on: Mar 10, 2015
 * Author: Karol Dzitkowski
 */

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
			("help", "produce help message")
			("c", "compress")
			("d", "decompress")
			("input-file", po::value<string>(), "input file")
			("output-file", po::value<string>(), "output file")
			;
	options.ConsoleOptions = consoleOptions;

	// SET CONSOLE POSITIONAL OPTIONS
	po::positional_options_description consolePositionalOptions("Console positional options");
	consolePositionalOptions.add("input-file", -2);
	consolePositionalOptions.add("output-file", -1);
	options.ConsolePositionalOptions = consolePositionalOptions;

	// SET CONFIG FILE OPTIONS
	po::options_description configFileOptions("Config file options");
	options.ConfigFileOptions = configFileOptions;

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
	ConfigDefinition configDef { argc, argv, "config.ini", GetProgramOptions() };
	ddj::Config::Initialize(configDef);

	initialize_logger();



	return 0;
}
