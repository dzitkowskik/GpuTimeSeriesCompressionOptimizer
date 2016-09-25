//
// Created by Karol Dzitkowski on 11.10.15.
//

#ifndef TIME_SERIES_DATA_READER_TIME_SERIES_H
#define TIME_SERIES_DATA_READER_TIME_SERIES_H

#include "column.hpp"
#include "ts_file_definition.hpp"

#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <boost/make_shared.hpp>

class TimeSeries;
using SharedTimeSeriesPtr = boost::shared_ptr<TimeSeries>;

class TimeSeries
{
public:
    TimeSeries() : _name(""), _recordsCnt(0) {}
    TimeSeries(std::string name) : _name(name), _recordsCnt(0) {}
    ~TimeSeries() {}
    TimeSeries(const TimeSeries& other)
            : _name(other._name), _columns(other._columns), _recordsCnt(other._recordsCnt)
    {}
    TimeSeries(TimeSeries&& other)
            : _name(std::move(other._name)),
              _columns(std::move(other._columns)),
              _recordsCnt(std::move(other._recordsCnt))
    {}

public:
    void init(std::vector<DataType> columnTypes)
    {
        for(auto& type : columnTypes)
            _columns.push_back(Column(type));
    }

    void setDecimals(std::vector<int> columnDecimals)
    {
        size_t len = std::min(columnDecimals.size(), _columns.size());
        for(int i = 0; i < len; i++)
            _columns[i].setDecimal(columnDecimals[i]);
    }

    Column& getColumn(size_t colIdx) { return _columns[colIdx]; }
    std::string getName() { return _name; }
    size_t getColumnsNumber() { return _columns.size(); }
    size_t getRecordsCnt() { return _recordsCnt; }
    void updateRecordsCnt() { _recordsCnt = _columns[0].getSize() / _columns[0].getDataSize(); }

    void setName(std::string name) { _name = name; }

    void setColumnNames(std::vector<std::string> names)
    {
        for(int i = 0; i < names.size(); i++)
            _columns[i].setName(names[i]);
    }

    std::vector<std::string> getColumnNames()
    {
        std::vector<std::string> result;
        for(auto& column : _columns)
            result.push_back(column.getName());
        return result;
    }

    size_t getRecordSize()
    {
        size_t size = 0;
        for(auto& column : _columns)
            size += column.getDataSize();
        return size;
    }

    void addRecord(char* data)
    {
        size_t offset = 0;
        for(auto& column : _columns)
            offset += column.addRawValue(data+offset);
        _recordsCnt++;
    }

    void addRecord(std::vector<std::string>& record)
    {
        if(record.size() != _columns.size())
            throw std::runtime_error(_recordSizeErrorMsg);

        for(int i = 0; i < record.size(); i++)
            _columns[i].addStringValue(record[i]);
        _recordsCnt++;
    }

    std::vector<RawData> getRawRecordData(size_t rowIdx)
    {
        std::vector<RawData> result;
        for(auto& column : _columns)
            result.push_back(column.getRaw(rowIdx));
        return result;
    }

    std::vector<std::string> getRecordAsStrings(size_t rowIdx)
    {
        std::vector<std::string> result;
        for(auto& column : _columns)
            result.push_back(column.getStringValue(rowIdx));
        return result;
    }

    bool compare(TimeSeries& other)
    {
        if(_recordsCnt != other.getRecordsCnt()) return false;
        for(int i = 0; i < getColumnsNumber(); i++)
            if(!_columns[i].compare(other.getColumn(i)))
                return false;
        return true;
    }

    void print(int n)
    {
    	printf("TimeSeries %s:\n", getName().c_str());
    	for(int i = 0; i < n && i < getRecordsCnt(); i++)
    	{
			printf("Row %d: ", i);
			for(auto& row : getRecordAsStrings(i))
				printf("%s, ", row.c_str());
			printf("\n");
    	}
    	printf("--------\n");
    }

//    SharedTimeSeriesPtr copy(int rows = 0)
//    {
//    	if(rows == 0) rows = getRecordsCnt();
//    	FileDefinition def;
//    	for(auto& column : _columns)
//    	{
//    		def.Columns.push_back(column.getType());
//    		def.Header.push_back(column.getName());
//    		def.Decimals.push_back(column.getDecimal());
//    	}
//    	auto result = TimeSeries::make_shared(def);
//    	for(int i = 0; i < rows; i++)
//    		result->addRecord(getRawRecordData(i).Data);
//    	return result;
//    }

public:
    static SharedTimeSeriesPtr make_shared(FileDefinition& def)
    {
    	auto ts = boost::make_shared<TimeSeries>();
    	ts->init(def.Columns);
    	ts->setColumnNames(def.Header);
    	return ts;
    }

private:
    std::string _name;
    std::vector<Column> _columns;
    size_t _recordsCnt;

private:
    const char* _recordSizeErrorMsg =
            "Record size does not equal number of columns";
};

#endif //TIME_SERIES_DATA_READER_TIME_SERIES_H
