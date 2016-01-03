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
		throw std::runtime_error("Config not initialized!");

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
