/*
 * data/time_series_reader.hpp
 *
 *  Created on: Dec 12, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef TIME_SERIES_READER_HPP_
#define TIME_SERIES_READER_HPP_

#include "data/time_series.hpp"
#include "data/file.hpp"
#include "ts_file_definition.hpp"

#include <boost/make_shared.hpp>

class TimeSeriesReader;
using SharedTimeSeriesReader = boost::shared_ptr<TimeSeriesReader>;

class TimeSeriesReader
{
public:
	TimeSeriesReader()
		: _lastFilePosition(0)
	{}
	virtual ~TimeSeriesReader() {}
	TimeSeriesReader(const TimeSeriesReader& other)
		: _lastFilePosition(other._lastFilePosition)
	{}

public:
	virtual SharedTimeSeriesPtr Read(File& file, const int maxRows = INT32_MAX) = 0;
	virtual void Write(File& file, TimeSeries& series) = 0;

public:
	static FileDefinition ReadFileDefinition(File& file);
	static void WriteFileDefinition(File& file, FileDefinition& definition);

protected:
	size_t _lastFilePosition;
};

class TimeSeriesReaderCSV : public TimeSeriesReader
{
public:
	TimeSeriesReaderCSV() {}
	TimeSeriesReaderCSV(CSVFileDefinition definition)
		: _definition(definition)
	{}
	virtual ~TimeSeriesReaderCSV() {}
	TimeSeriesReaderCSV(const TimeSeriesReaderCSV& other)
		: _definition(other._definition)
	{}

public:
	SharedTimeSeriesPtr Read(File& file,	const int maxRows = INT32_MAX);
	void Write(File& file, TimeSeries& series);

public:
	static boost::shared_ptr<TimeSeriesReaderCSV> make_shared(CSVFileDefinition definition)
	{
		return boost::make_shared<TimeSeriesReaderCSV>(definition);
	}

private:
	CSVFileDefinition _definition;
};


class TimeSeriesReaderBinary : public TimeSeriesReader
{
public:
	TimeSeriesReaderBinary() : _alignment(0) {}
	TimeSeriesReaderBinary(BinaryFileDefinition definition, size_t alignment = 0)
			: _definition(definition), _alignment(alignment)
	{}
	virtual ~TimeSeriesReaderBinary() {}
	TimeSeriesReaderBinary(const TimeSeriesReaderBinary& other)
		: _definition(other._definition), _alignment(other._alignment)
	{}

public:
	SharedTimeSeriesPtr Read(File& file, const int maxRows = INT32_MAX);
	void Write(File& file, TimeSeries& series);

public:
	static boost::shared_ptr<TimeSeriesReaderBinary> make_shared(BinaryFileDefinition definition, size_t alignment = 0)
	{
		return boost::make_shared<TimeSeriesReaderBinary>(definition, alignment);
	}

private:
	BinaryFileDefinition _definition;
	size_t _alignment;
};

#endif /* TIME_SERIES_READER_HPP_ */
