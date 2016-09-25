/*
 * enum_parser.hpp
 *
 *  Created on: Jan 3, 2016
 *      Author: Karol Dzitkowski
 */

#ifndef ENUM_PARSER_HPP_
#define ENUM_PARSER_HPP_

#include <map>
#include <string>

template <typename T>
class EnumParser
{
public:
    EnumParser(){};

    T Parse(const std::string &value)
    {
        auto iValue = _enumMap.find(value);
        if (iValue  == _enumMap.end())
            throw std::runtime_error("Enum value cannot be parsed!");
        return iValue->second;
    }
private:
    std::map<std::string, T> _enumMap;
};

#endif /* ENUM_PARSER_HPP_ */
