/*
 * parallel_ts_compressor.hpp
 *
 *  Created on: Dec 29, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef PARALLEL_TS_COMPRESSOR_HPP_
#define PARALLEL_TS_COMPRESSOR_HPP_

#include "core/task/task_scheduler.hpp"
#include "time_series_reader.hpp"

namespace ddj
{

class ParallelTSCompressor
{
public:
	ParallelTSCompressor();
	virtual ~ParallelTSCompressor();

public:
	TimeSeries GetNextBatch();
	void Compress();

private:
	UniqueTaskSchedulerPtr _taskScheduler;
	size_t _batchSize;
	size_t _probeSize;
	int _columnNumber;
	std::vector<SharedTimeSeriesPtr> _timeSeriesToCompress;
};

} /* namespace ddj */

#endif /* PARALLEL_TS_COMPRESSOR_HPP_ */
