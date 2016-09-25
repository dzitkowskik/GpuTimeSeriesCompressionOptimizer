/*
 * time_series_reader.cpp
 *
 *  Created on: Dec 12, 2015
 *      Author: Karol Dzitkowski
 */

#include "data/time_series_reader.hpp"
#include <iomanip>
#include <boost/lexical_cast.hpp>

std::vector<std::string> ReadHeader(std::ifstream& inFile, CSVFileDefinition& definition)
{
    std::vector<std::string> result;
    std::string headerLine;
    std::getline(inFile, headerLine);
    size_t position = 0;
    do
    {
        position = headerLine.find(definition.Separator);
        auto colName = headerLine.substr(0, position);
        headerLine.erase(0, position + definition.Separator.length());
        result.push_back(colName);
    } while(std::string::npos != position);

    definition.Header = result;

    printf("%s\n", "Header read success");
    return result;
}

SharedTimeSeriesPtr TimeSeriesReaderCSV::Read(
        File& file, const int maxRows)
{
    // Initialize time series
    auto result = boost::make_shared<TimeSeries>(file.GetPath());
    result->init(_definition.Columns);
    result->setDecimals(_definition.Decimals);

    // Open file and set last position
    std::ifstream inputFile(file.GetPath(), std::ios::in);
    inputFile.seekg(_lastFilePosition, inputFile.beg);

    // Read header
    std::vector<std::string> header;
    if(_definition.HasHeader && _lastFilePosition == 0)
        header = ReadHeader(inputFile, _definition);
    else
        header = _definition.Header;
    result->setColumnNames(header);

    std::string line, token;
    size_t position = 0;
    int count = 0;
    while((count++ < maxRows) && std::getline(inputFile, line))
    {
        std::vector<std::string> record;
        do
        {
            position = line.find(_definition.Separator);
            token = line.substr(0, position);
            line.erase(0, position + _definition.Separator.length());
            record.push_back(token);
        } while(position != std::string::npos);
        result->addRecord(record);
        record.clear();
    }

    _lastFilePosition = inputFile.tellg();
    inputFile.close();

    return result;
}

SharedTimeSeriesPtr TimeSeriesReaderBinary::Read(
        File& file,
        const int maxRows)
{
    // Initialize time series
    auto result = boost::make_shared<TimeSeries>(file.GetPath());
    result->init(_definition.Columns);
    result->setDecimals(_definition.Decimals);

    size_t size = result->getRecordSize();
    size += _alignment;
//    printf("size = %lu\n", size);
    result->setColumnNames(_definition.Header);

    char* data = new char[size];
    int count = 0;

    while((count++ < maxRows) && (-1 != file.ReadRaw(data, size)))
        result->addRecord(data);

    delete [] data;
    return result;
}

void WriteLine(
        std::ofstream& ofstream,
        std::vector<std::string> lineItems,
        CSVFileDefinition& definition)
{
    std::string line;
    for(int i = 0; i < lineItems.size(); i++)
    {
        line += lineItems[i];
        if(i < lineItems.size()-1) line += definition.Separator;
    }
    ofstream << line << std::endl;
}

void TimeSeriesReaderCSV::Write(File& file, TimeSeries& series)
{
    std::ofstream outFile(file.GetPath(), std::ios::app);

    // Write header as column names
    if(_definition.HasHeader && _lastFilePosition == 0)
        WriteLine(outFile, series.getColumnNames(), _definition);

    for(size_t i = 0; i < series.getRecordsCnt(); i++)
        WriteLine(outFile, series.getRecordAsStrings(i), _definition);

    outFile.close();
}

void TimeSeriesReaderBinary::Write(File& file, TimeSeries& series)
{
    size_t size = series.getRecordSize() + _alignment;
    char* data = new char[size];

    for(size_t i = 0; i < series.getRecordsCnt(); i++)
    {
        size_t offset = 0;
        memset(data, 0, size);
        for(auto& rawData : series.getRawRecordData(i))
        {
            memcpy(data+offset, rawData.Data, rawData.Size);
            offset += rawData.Size;
        }
        if (file.WriteRaw(data, size))
            throw std::runtime_error("Error while writting to a file");
    }

    delete [] data;
}

FileDefinition TimeSeriesReader::ReadFileDefinition(File& file)
{
	FileDefinition result;
	std::ifstream inputFile(file.GetPath(), std::ios::in);
	std::string line, name, type, decimal;
	EnumParser<DataType> typeParser;
    size_t position = 0;
	while(std::getline(inputFile, line))
	{
        // GET NAME
        position = line.find(',');
        name = line.substr(0, position);
        line.erase(0, position + 1);

        // GET TYPE
        position = line.find(',');
        type = line.substr(0, position);
        line.erase(0, position + 1);

        // GET DECIMAL
        decimal = line;

		result.Header.push_back(name);
		result.Columns.push_back(typeParser.Parse(type));
        result.Decimals.push_back(boost::lexical_cast<int>(decimal));
	}

	inputFile.close();
	return result;
}

void TimeSeriesReader::WriteFileDefinition(File& file, FileDefinition& definition)
{
	std::ofstream outFile(file.GetPath(), std::ios::out);

	for(int i = 0; i < definition.Columns.size(); i++)
	{
		outFile << definition.Header[i] << ',';
		outFile << GetDataTypeString(definition.Columns[i]) << ',';
		if(definition.Decimals.size() > i)
			outFile << definition.Decimals[i];
		else outFile << "0";
		outFile << std::endl;
	}

	outFile.close();
}
