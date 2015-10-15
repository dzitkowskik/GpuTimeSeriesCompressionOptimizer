//
// Created by Karol Dzitkowski on 09.10.15.
//

#include "time_series_file_reader.h"
#include "helpers/helper_macros.h"
#include <sstream>
#include <iostream>

#ifdef DDJ_TIME_SERIES_READER_USE_BOOST
#include <boost/lexical_cast.hpp>
#endif

std::vector<std::string> TimeSeriesFileReader::ReadHeader(std::ifstream& inFile)
{
    std::vector<std::string> result;
    std::string headerLine;
    std::getline(inFile, headerLine);
    size_t position = 0;
    do
    {
        position = headerLine.find(this->_definition.Separator);
        auto colName = headerLine.substr(0, position);
        headerLine.erase(0, position + this->_definition.Separator.length());
        result.push_back(colName);
    } while(std::string::npos != position);

    return result;
}

#ifndef DDJ_TIME_SERIES_READER_USE_BOOST
template<typename T>
T ParseString(std::string value)
{
    std::istringstream stream(value);
    T result;
    stream >> result;
    return result;
}
#else
template<typename T>
T ParseString(std::string value)
{
    return boost::lexical_cast<T>(value);
}
#endif

template<typename DataType, typename TimeType>
std::vector<TimeSeries<DataType, TimeType>> TimeSeriesFileReader::Read(std::string path)
{
    std::vector<TimeSeries<DataType, TimeType>> result;
    std::ifstream inputFile(path, std::ios::in);

    // GET HEADER
    std::vector<std::string> header;
    if(this->_definition.UseCustomHeader)
        header = this->_definition.Header;
    else
        header = this->ReadHeader(inputFile);

    // Create time series with names from header
    for(int index = 0; index < header.size(); index++)
        if(index != this->_definition.TimeValueIndex)
            result.push_back(TimeSeries<DataType, TimeType>(header[index]));

    std::string line, token;
    size_t position = 0, i, j;

    while(std::getline(inputFile, line))
    {
        i = 0, j = 0;  // column number
        do
        {
            position = line.find(this->_definition.Separator);
            token = line.substr(0, position);
            line.erase(0, position + this->_definition.Separator.length());

            if(j++ == this->_definition.TimeValueIndex)
                for(auto& ts : result) ts.InsertTime(ParseString<TimeType>(token));
            else
                result[i++].InsertData(ParseString<DataType>(token));
        } while(position != std::string::npos);
    }

    inputFile.close();
    return result;
}

#define READER_DATA_TYPE_SPEC(X) \
	template std::vector<TimeSeries<X, int>> TimeSeriesFileReader::Read<X, int>(std::string path); \
    template std::vector<TimeSeries<X, unsigned int>> TimeSeriesFileReader::Read<X, unsigned int>(std::string path); \
    template std::vector<TimeSeries<X, long long int>> TimeSeriesFileReader::Read<X, long long int>(std::string path); \
    template std::vector<TimeSeries<X, unsigned long long int>> TimeSeriesFileReader::Read<X, unsigned long long int>(std::string path); \
    template std::vector<TimeSeries<X, long int>> TimeSeriesFileReader::Read<X, long int>(std::string path); \
    template std::vector<TimeSeries<X, unsigned long int>> TimeSeriesFileReader::Read<X, unsigned long int>(std::string path);
FOR_EACH(READER_DATA_TYPE_SPEC, double, float, int, unsigned int, long, unsigned long, long long, unsigned long long)
