/*
 * parallel_ts_compressor.cpp
 *
 *  Created on: Dec 29, 2015
 *      Author: Karol Dzitkowski
 */

#include "parallel_ts_compressor.hpp"
#include "optimizer/compression_task.hpp"
#include <queue>
#include <boost/make_shared.hpp>

namespace ddj
{

ParallelTSCompressor::ParallelTSCompressor(SharedTimeSeriesReader reader)
	: _reader(reader),
	  _batchSize(10e5),
	  _columnNumber(2),
	  _initialized(false)
{
}

void ParallelTSCompressor::init(SharedTimeSeriesPtr ts)
{
	_columnNumber = ts->getColumnsNumber();
	for(int i = 0; i < _columnNumber; i++)
		_optimizers.push_back(CompressionOptimizer::make_shared());
	_taskScheduler = TaskScheduler::make_unique(_columnNumber);
	_initialized = true;
}

void ParallelTSCompressor::Compress(File& inputFile, File& outputFile)
{
	SharedTimeSeriesPtr ts;
	do
	{
		// read batch data from source
		ts = _reader->Read(inputFile, _batchSize);

		// if not initialized init optimizers for every column
		if(!_initialized) init(ts);

		// schedule tasks for compression of new column parts
		for(int i = 0; i < _columnNumber; i++)
			_taskScheduler->Schedule(CompressionTask::make_shared(ts, i, _optimizers[i], outputFile));

		// wait for all tasks to complete
		_taskScheduler->WaitAll();
		_taskScheduler->Clear();
	} while(ts->getRecordsCnt() == _batchSize);

}

} /* namespace ddj */
