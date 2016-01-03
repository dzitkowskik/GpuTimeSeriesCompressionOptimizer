/*
 * Config.hpp
 *
 *  Created on: Mar 13, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_CONFIG_H_
#define DDJ_CONFIG_H_

#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;

namespace ddj
{

struct ConfigOptions
{
	po::options_description ConfigFileOptions;
	po::options_description ConsoleOptions;
	po::positional_options_description ConsolePositionalOptions;
};

struct ConfigDefinition
{
	int argc;
	char** argv;
	std::string ConfigFile;
	ConfigOptions Options;
};

class Config
{
private:
    static Config* _instance;
    po::variables_map _configMap;

    Config(ConfigDefinition definition);
    virtual ~Config() {};

public:
    template<typename T> T GetValue(string settingName)
    {
        if (_configMap.count(settingName))
            return _configMap[settingName].as<T>();
        return T();
    }

    bool HasValue(string settingName);
    void ListAllSettings();

public:
    static Config* GetInstance();
    static void Initialize(ConfigDefinition definition);
};

} /* namespace ddj */
#endif /* DDJ_CONFIG_H_ */
