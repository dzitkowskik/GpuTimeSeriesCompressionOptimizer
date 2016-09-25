/*
 *  data/data_type.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#pragma once
#ifndef DDJ_DATA_TYPE_HPP_
#define DDJ_DATA_TYPE_HPP_

#include "enum_parser.hpp"

#include <cstring>
#include <time.h>
#include <string>

enum class DataType
{
    d_time,
    d_char,
    d_int,
    d_unsigned,
    d_float,
    d_double,
    d_boolean,
    d_short
};

inline size_t GetDataTypeSize(DataType type)
{
    switch(type)
    {
        case DataType::d_time: return sizeof(time_t);
        case DataType::d_char: return sizeof(char);
        case DataType::d_int: return sizeof(int);
        case DataType::d_unsigned: return sizeof(unsigned);
        case DataType::d_float: return sizeof(float);
        case DataType::d_double: return sizeof(double);
        case DataType::d_boolean: return sizeof(bool);
        case DataType::d_short: return sizeof(short);
    }
    return 0;
}

template<typename T> inline DataType GetDataType();
template<> inline DataType GetDataType<time_t>() { return DataType::d_time; }
template<> inline DataType GetDataType<char>() { return DataType::d_char; }
template<> inline DataType GetDataType<int>() { return DataType::d_int; }
template<> inline DataType GetDataType<unsigned>() { return DataType::d_unsigned; }
template<> inline DataType GetDataType<float>() { return DataType::d_float; }
template<> inline DataType GetDataType<double>() { return DataType::d_double; }
template<> inline DataType GetDataType<bool>() { return DataType::d_boolean; }
template<> inline DataType GetDataType<short>() { return DataType::d_short; }

template<> inline
EnumParser<DataType>::EnumParser()
{
    _enumMap["time"] = DataType::d_time;
    _enumMap["char"] = DataType::d_char;
    _enumMap["int"] = DataType::d_int;
    _enumMap["unsigned"] = DataType::d_unsigned;
    _enumMap["float"] = DataType::d_float;
    _enumMap["double"] = DataType::d_double;
    _enumMap["bool"] = DataType::d_boolean;
    _enumMap["short"] = DataType::d_short;
}

inline std::string GetDataTypeString(DataType type)
{
    switch(type)
    {
        case DataType::d_time: return "time_t";
        case DataType::d_char: return "char";
        case DataType::d_int: return "int";
        case DataType::d_unsigned: return "unsigned int";
        case DataType::d_float: return "float";
        case DataType::d_double: return "double";
        case DataType::d_boolean: return "bool";
        case DataType::d_short: return "short";
    }
    return 0;
}

#endif /* DDJ_DATA_TYPE_HPP_ */
