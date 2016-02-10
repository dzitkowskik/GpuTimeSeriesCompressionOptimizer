/*
 * compression_task.hpp
 *
 *  Created on: Dec 30, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef COMPRESSION_TASK_HPP_
#define COMPRESSION_TASK_HPP_

#include "file.hpp"
#include "core/task/task.hpp"
#include "core/logger.h"
#include "time_series.hpp"
#include "optimizer/compression_optimizer.hpp"
#include <boost/make_shared.hpp>

namespace ddj
{

class CompressionTask;
using SharedCompressionTaskPtr = boost::shared_ptr<CompressionTask>;

class CompressionTask : public Task
{
public:
	CompressionTask(SharedTimeSeriesPtr ts, int columnId, SharedCompressionOptimizerPtr optimizer, File outputFile)
		: _ts(ts),
		  _columnId(columnId),
		  _optimizer(optimizer),
		  _deviceId(0),
		  _outputFile(outputFile),
		  _logger(log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("CompressionTask")))
	{}

	virtual ~CompressionTask() {}

	CompressionTask(const CompressionTask& other)
		: _ts(other._ts),
		  _columnId(other._columnId),
		  _optimizer(other._optimizer),
		  _deviceId(other._deviceId),
		  _outputFile(other._outputFile),
		  _logger(log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("CompressionTask")))
	{}

public:
	void SetDevice(int deviceId) { _deviceId = deviceId; }

public:
	static SharedCompressionTaskPtr make_shared(
			SharedTimeSeriesPtr ts,
			int columnId,
			SharedCompressionOptimizerPtr optimizer,
			File outputFile)
	{
		return boost::make_shared<CompressionTask>(ts, columnId, optimizer, outputFile);
	}

protected:
	void execute();

private:
	int _deviceId;
	int _columnId;
	SharedTimeSeriesPtr _ts;
	SharedCompressionOptimizerPtr _optimizer;
	File _outputFile;
	log4cplus::Logger _logger;
};

} /* namespace ddj */

#endif /* COMPRESSION_TASK_HPP_ */
