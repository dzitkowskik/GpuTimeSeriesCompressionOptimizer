/*
 * parallel_ts_compressor.hpp
 *
 *  Created on: Dec 29, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef PARALLEL_TS_COMPRESSOR_HPP_
#define PARALLEL_TS_COMPRESSOR_HPP_

#include "optimizer/compression_optimizer.hpp"
#include "core/task/task_scheduler.hpp"
#include "data/time_series_reader.hpp"
#include "core/logger.h"

namespace ddj
{

class ParallelTSCompressor
{
public:
	ParallelTSCompressor(SharedTimeSeriesReader reader);
	virtual ~ParallelTSCompressor() {}

public:
	void Compress(File& inputFile, File& outputFile);
	void Decompress(File& inputFile, File& outputFile, FileDefinition& def);
	void SetBatchSize(size_t size);

private:
	void init(SharedTimeSeriesPtr ts);

private:
	bool _initialized;
	size_t _batchSize;
	int _columnNumber;
	log4cplus::Logger _logger;

	UniqueTaskSchedulerPtr _taskScheduler;
	SharedTimeSeriesReader _reader;
	std::vector<SharedTimeSeriesPtr> _timeSeriesToCompress;
	std::vector<SharedCompressionOptimizerPtr> _optimizers;
};

} /* namespace ddj */

#endif /* PARALLEL_TS_COMPRESSOR_HPP_ */
