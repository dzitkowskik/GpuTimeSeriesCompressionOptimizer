/*
* Config.cpp
*
*  Created on: Mar 13, 2015
*      Author: Karol Dzitkowski
*/

#include "config.hpp"
#include <iostream>
#include <fstream>
#include <iterator>



namespace ddj
{

Config* Config::_instance(0);

ConfigOptions GetDefaultProgramOptions()
{
	ConfigOptions options;

	// SET CONSOLE OPTIONS
	po::options_description consoleOptions("Config file options");
	options.ConsoleOptions.add(consoleOptions);

	// SET CONSOLE POSITIONAL OPTIONS
	po::positional_options_description consolePositionalOptions;
	options.ConsolePositionalOptions = consolePositionalOptions;

	// SET CONFIG FILE OPTIONS
	po::options_description configFileOptions("Config file options");
	configFileOptions.add_options()
	    ("TEST_DATA_LOG", po::value<std::string>()->default_value(""), "default file containing test time series data")
	    ("BENCHMARK_DATA_LOG", po::value<std::string>()->default_value(""), "default file containing benchmark time series data")
	    ("NYSE_DATA_1GB", po::value<std::string>()->default_value(""), "default file containing nyse time series data from openbook")
		("LOG_CONFIG", po::value<std::string>()->default_value("logger.prop", "file containing log4cplus configuration"))
	    ;
	options.ConfigFileOptions.add(configFileOptions);

	return options;
}

Config::Config(ConfigDefinition definition)
{
	try
	{
		_configMap = po::variables_map();

		definition.Options.ConsoleOptions.add_options()
		("help", "shows help message")
		;

		if(definition.ConfigFile.size() > 0)
		{
			ifstream ifs(definition.ConfigFile.c_str());
			if (!ifs)
			{
				string msg = "can not open config file: ";
				msg.append(definition.ConfigFile);
				fprintf(stderr, "Error in %s: %s\n", __FILE__, msg.c_str());
			}
			else
				po::store(po::parse_config_file(ifs, definition.Options.ConfigFileOptions), _configMap);
		}

		auto commandLineParser = po::command_line_parser(definition.argc, definition.argv)
			.options(definition.Options.ConsoleOptions)
			.positional(definition.Options.ConsolePositionalOptions)
			.run();
		po::store(commandLineParser, _configMap);
		notify(_configMap);

		if (_configMap.count("help"))
		{
			std::cout << definition.Options.ConsoleOptions << std::endl;
			exit(0);
		}
	}
	catch (exception& e)
	{
		fprintf(stderr, "Error in file %s: %s\n", __FILE__, e.what());
	}
}

Config* Config::GetInstance()
{
	if (!_instance)
		Initialize(ConfigDefinition{0, nullptr, "config.ini", GetDefaultProgramOptions()});

	return _instance;
}

bool Config::HasValue(string settingName)
{
	return _configMap.count(settingName);
}

void Config::Initialize(ConfigDefinition definition)
{
	if (!_instance)
	{
		_instance = new Config(definition);
	}
}

void Config::ListAllSettings()
{
	po::variables_map::iterator it;
	for (it = _configMap.begin(); it != _configMap.end(); ++it)
		cout << it->first << "\n";
}


} /* namespace ddj */
