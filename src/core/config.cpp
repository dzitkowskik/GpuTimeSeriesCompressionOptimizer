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

  Config* Config::GetInstance()
  {
    if (!_instance)
    {
      _instance = new Config;
    }

    return _instance;
  }

  bool Config::HasValue(string settingName)
  {
    return _configMap.count(settingName);
  }

  void Config::InitOptions(int argc, char** argv, string path)
  {
    try
    {
      _configMap = po::variables_map();

      po::options_description console_options("Allowed options");
      console_options.add_options()
          ("help", "shows help message")
          ("test", "runs all unit tests")
          ("performance", "runs all performance tests")
      ;

      po::options_description hidden("Hidden options");
      hidden.add_options()
      ("TEST_DATA_LOG", po::value<std::string>()->default_value(""), "default file containing test time series data")
      ;

      ifstream ifs(path.c_str());
      if (!ifs)
      {
        string msg = "can not open config file: ";
        msg.append(path);
        fprintf(stderr, "Error in %s: %s\n", __FILE__, msg.c_str());
        return;
      }
      else
      {
        po::store(po::parse_config_file(ifs, hidden), _configMap);
        po::store(po::parse_command_line(argc, argv, console_options), _configMap);
        notify(_configMap);
      }

      if (_configMap.count("help"))
      {
          std::cout << console_options << std::endl;
          exit(0);
      }
    }
    catch (exception& e)
    {
      fprintf(stderr, "Error in file %s: %s\n", __FILE__, e.what());
    }
  }

  void Config::ListAllSettings()
  {
    po::variables_map::iterator it;
    for (it = _configMap.begin(); it != _configMap.end(); ++it)
    {
      cout << it->first << "\n";
    }
  }
} /* namespace ddj */
