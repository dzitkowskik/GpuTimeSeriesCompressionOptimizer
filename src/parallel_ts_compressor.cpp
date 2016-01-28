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
	  _initialized(false)
{
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
	CUDA_CALL( cudaGetLastError() );

	SharedTimeSeriesPtr ts;
	do
	{
		// read batch data from source
		ts = _reader->Read(inputFile, _batchSize);
		ts->print(5);

		if(!_initialized) init(ts);

		// schedule tasks for compression of new column parts
		for(int i = 0; i < _columnNumber; i++)
			_taskScheduler->Schedule(CompressionTask::make_shared(ts, i, _optimizers[i], outputFile));

		// wait for all tasks to complete
		_taskScheduler->WaitAll();
		_taskScheduler->Clear();
	} while(ts->getRecordsCnt() == _batchSize);
}

void ParallelTSCompressor::Decompress(File& inputFile, File& outputFile, FileDefinition& def)
{
	CUDA_CALL( cudaGetLastError() );

	// Create part of time series
	SharedTimeSeriesPtr ts = TimeSeries::make_shared(def);

	// read size of data chunk
	size_t size = 0;
	int i = 0;
	// printf("Start decompressing!\n");
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
			i = 0;
			// wait for all tasks to complete
			_taskScheduler->WaitAll();
			_taskScheduler->Clear();
			// write part of time series to file
			ts->updateRecordsCnt();
			_reader->Write(outputFile, *ts);
		}
	}
}

} /* namespace ddj */
