/*
 * parallel_ts_compressor.cpp
 *
 *  Created on: Dec 29, 2015
 *      Author: Karol Dzitkowski
 */

#include "parallel_ts_compressor.hpp"
#include "optimizer/compression_task.hpp"
#include "optimizer/decompression_task.hpp"
#include <queue>
#include <boost/make_shared.hpp>

namespace ddj
{

ParallelTSCompressor::ParallelTSCompressor(SharedTimeSeriesReader reader)
	: _reader(reader),
	  _batchSize(1e6),
	  _columnNumber(2),
	  _initialized(false),
	  _logger(log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("ParallelTSCompressor")))
{}

void ParallelTSCompressor::SetBatchSize(size_t size)
{
	_batchSize = size;
}

void ParallelTSCompressor::init(SharedTimeSeriesPtr ts)
{
	_columnNumber = ts->getColumnsNumber();
	for(int i = 0; i < _columnNumber; i++)
		_optimizers.push_back(CompressionOptimizer::make_shared());
	_taskScheduler = TaskScheduler::make_unique(1);
	_initialized = true;
}

void ParallelTSCompressor::Compress(File& inputFile, File& outputFile)
{
	LOG4CPLUS_DEBUG(_logger, "Starting compression");
	CUDA_CALL( cudaGetLastError() );

	SharedTimeSeriesPtr ts;
	do
	{
		// read batch data from source
		ts = _reader->Read(inputFile, _batchSize);
//		ts->print(5);
		if(ts->getRecordsCnt() <= 0) break;

		if(!_initialized) init(ts);

		// schedule tasks for compression of new column parts
		for(int i = 0; i < _columnNumber; i++)
			_taskScheduler->Schedule(CompressionTask::make_shared(ts, i, _optimizers[i], outputFile));

		// wait for all tasks to complete
		_taskScheduler->WaitAll();
		_taskScheduler->Clear();
	} while(ts->getRecordsCnt() == _batchSize);
	LOG4CPLUS_DEBUG(_logger, "Compression done");
}

void ParallelTSCompressor::Decompress(File& inputFile, File& outputFile, FileDefinition& def)
{
	LOG4CPLUS_DEBUG(_logger, "Starting decompression");
	CUDA_CALL( cudaGetLastError() );

	// Create part of time series
	SharedTimeSeriesPtr ts = TimeSeries::make_shared(def);

	// read size of data chunk
	size_t size = 0;
	int i = 0;
	while(!inputFile.ReadRaw((char*)&size, sizeof(size_t)))
	{
		if(!_initialized) init(ts);

		// read data from file and save as ts column
		ts->getColumn(i).reserveSize(size);
		inputFile.ReadRaw(ts->getColumn(i).getData(), size);

		// schedule decompression task
		_taskScheduler->Schedule(DecompressionTask::make_shared(ts, i++));
		if(i == _columnNumber)
		{
			LOG4CPLUS_DEBUG(_logger, "SYNC!");
			i = 0;

			// wait for all tasks to complete
			_taskScheduler->WaitAll();
			_taskScheduler->Clear();

			// write part of time series to file
			ts->updateRecordsCnt();
			_reader->Write(outputFile, *ts);
			LOG4CPLUS_DEBUG_FMT(_logger, "%d records written to a file", ts->getRecordsCnt());
		}
	}
	LOG4CPLUS_DEBUG(_logger, "Decompression done");
}

} /* namespace ddj */
