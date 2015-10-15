//
// Created by Karol Dzitkowski on 09.10.15.
//

#ifndef TIME_SERIES_DATA_READER_READER_H
#define TIME_SERIES_DATA_READER_READER_H

#include <string>
#include <vector>
#include <fstream>
#include "time_series.h"

struct InputDefinition
{
    std::vector<std::string> Header;
    bool UseCustomHeader = false;
    int TimeValueIndex = 0;
    std::string Separator = ",";
};

class TimeSeriesFileReader
{
public:
    TimeSeriesFileReader(InputDefinition inputDefinition)
            : _definition(inputDefinition)
    {}
    ~TimeSeriesFileReader() {}
    TimeSeriesFileReader(const TimeSeriesFileReader& other)
            : _definition(other._definition)
    {}
    TimeSeriesFileReader(TimeSeriesFileReader&& other)
            : _definition(std::move(other._definition))
    {}

public:
    template<typename DataType = float, typename TimeType = unsigned long long int>
    std::vector<TimeSeries<DataType, TimeType>> Read(std::string path);

private:
    std::vector<std::string> ReadHeader(std::ifstream& inFile);

private:
    InputDefinition _definition;
};


#endif //TIME_SERIES_DATA_READER_READER_H
