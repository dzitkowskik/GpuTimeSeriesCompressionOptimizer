/*
 * Config.cpp
 *
 *  Created on: Mar 13, 2015
 *      Author: Karol Dzitkowski
 */

#include "Config.h"
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
      ("MB_SIZE_IN_BYTES", po::value<int>()->default_value(1048576), "size in bytes")
      ("MAIN_STORE_SIZE", po::value<int>()->default_value(512*1048576), "main store size")
      ("GPU_MEMORY_ALLOC_ATTEMPTS", po::value<int>()->default_value(8), "number of GPU memory allocation attempts")
      ("STREAMS_NUM_QUERY", po::value<int>()->default_value(4), "number of query streams")
      ("STREAMS_NUM_UPLOAD", po::value<int>()->default_value(4), "number of upload streams")
      ("STORE_BUFFER_CAPACITY", po::value<int>()->default_value(512), "store buffer capacity")
      ("INSERT_THREAD_POOL_SIZE", po::value<int>()->default_value(2), "number of threads in thread pool for inserts")
      ("SELECT_THREAD_POOL_SIZE", po::value<int>()->default_value(6), "number of threads in thread pool for selects")
      ("SIMUL_QUERY_COUNT", po::value<int>()->default_value(4), "number of simultaneous queries")
      ("MASTER_IP_ADDRESS", po::value<string>()->default_value("127.0.0.1"), "address of master server")
      ("MASTER_LOGIN_PORT", po::value<string>()->default_value("8080"), "port of master server login service")
      ("ENABLE_COMPRESSION", po::value<int>()->default_value(1), "1 if compression enabled, 0 otherwise")
      ("MAX_JOB_MEMORY_SIZE", po::value<int>()->default_value(61440), "Maximal size of data used in one query job")
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
