//
// Created by Karol Dzitkowski on 11.10.15.
//

#ifndef TIME_SERIES_DATA_READER_WRITTER_H
#define TIME_SERIES_DATA_READER_WRITTER_H

#include <string>
#include <vector>
#include "time_series.h"

struct OutputDefinition
{
    std::vector<std::string> Header;
    bool UseCustomHeader = false;
    int TimeValueIndex = 0;
    std::string Separator = ",";
};

class TimeSeriesFileWriter
{
public:
    TimeSeriesFileWriter(OutputDefinition inputDefinition)
    : _definition(inputDefinition)
            {}
    ~TimeSeriesFileWriter() {}
    TimeSeriesFileWriter(const TimeSeriesFileWriter& other)
            : _definition(other._definition)
    {}
    TimeSeriesFileWriter(TimeSeriesFileWriter&& other)
    : _definition(std::move(other._definition))
    {}

public:
    template<typename DataType = float, typename TimeType = unsigned long long int>
    void Write(std::string path, std::vector<TimeSeries<DataType, TimeType>> series);

private:
    void WriteLine(std::ofstream&, std::vector<std::string>);

private:
    OutputDefinition _definition;
};


#endif //TIME_SERIES_DATA_READER_WRITTER_H
