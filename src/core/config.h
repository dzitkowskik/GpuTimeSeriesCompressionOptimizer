/*
 * Config.h
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

class Config
{
private:
    static Config* _instance;
    po::variables_map _configMap;

    Config() {};
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
    void InitOptions(int argc, char** argv, string path);

public:
    static Config* GetInstance();
};

} /* namespace ddj */
#endif /* DDJ_CONFIG_H_ */
