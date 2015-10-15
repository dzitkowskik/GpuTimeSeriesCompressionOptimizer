//
// Created by Karol Dzitkowski on 11.10.15.
//

#include "time_series_file_writer.h"
#include "helpers/helper_macros.h"
#include <fstream>
#include <iostream>
#include <sstream>

#ifdef DDJ_TIME_SERIES_READER_USE_BOOST
#include <boost/lexical_cast.hpp>
#endif

#ifndef DDJ_TIME_SERIES_READER_USE_BOOST
template<typename T>
std::string ParseType(T value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}
#else
template<typename T>
std::string ParseType(T value)
{
    return boost::lexical_cast<std::string>(value);
}
#endif

void TimeSeriesFileWriter::WriteLine(std::ofstream& ofstream, std::vector<std::string> lineItems)
{
    std::string line;
    for(int i = 0; i < lineItems.size(); i++)
    {
        line += lineItems[i];
        if(i < lineItems.size()-1) line += this->_definition.Separator;
    }
    ofstream << line << std::endl;
}

template<typename DataType, typename TimeType>
void TimeSeriesFileWriter::Write(std::string path, std::vector<TimeSeries<DataType, TimeType>> series)
{
    std::ofstream outFile(path, std::ios::out);

    //WRITE HEADER
    std::vector<std::string> header;
    if(this->_definition.UseCustomHeader)
        header = this->_definition.Header;
    else
    {
        for (auto &ts : series)
            header.push_back(ts.GetName());
        header.insert(header.begin()+this->_definition.TimeValueIndex, "time");
    }
    WriteLine(outFile, header);

    //WRITE DATA
    std::vector<std::string> lineItems;
    std::vector<typename std::vector<DataType>::iterator> dataIterators;
    for(auto& ts : series) dataIterators.push_back(ts.BeginData());

    for(auto timeIt = series[0].BeginTime(); timeIt != series[0].EndTime(); timeIt++)
    {
        lineItems.clear();
        for(int i = 0; i < dataIterators.size(); i++)
            lineItems.push_back(ParseType<DataType>(*dataIterators[i]++));
        lineItems.insert(lineItems.begin()+this->_definition.TimeValueIndex, ParseType<TimeType>(*timeIt));
        WriteLine(outFile, lineItems);
    }

    outFile.close();
}

#define WRITER_DATA_TYPE_SPEC(X) \
	template void TimeSeriesFileWriter::Write<X, int>(std::string, std::vector<TimeSeries<X, int>>); \
	template void TimeSeriesFileWriter::Write<X, unsigned int>(std::string, std::vector<TimeSeries<X, unsigned int>>); \
	template void TimeSeriesFileWriter::Write<X, long long int>(std::string, std::vector<TimeSeries<X, long long int>>); \
	template void TimeSeriesFileWriter::Write<X, unsigned long long int>(std::string, std::vector<TimeSeries<X, unsigned long long int>>); \
    template void TimeSeriesFileWriter::Write<X, long int>(std::string, std::vector<TimeSeries<X, long int>>); \
	template void TimeSeriesFileWriter::Write<X, unsigned long int>(std::string, std::vector<TimeSeries<X, unsigned long int>>);
FOR_EACH(WRITER_DATA_TYPE_SPEC, double, float, int, unsigned int, long, unsigned long, long long, unsigned long long)

