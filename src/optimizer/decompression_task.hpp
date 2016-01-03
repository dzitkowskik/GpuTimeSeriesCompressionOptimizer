/*
 * decompression_task.hpp
 *
 *  Created on: Jan 3, 2016
 *      Author: Karol Dzitkowski
 */

#ifndef DECOMPRESSION_TASK_HPP_
#define DECOMPRESSION_TASK_HPP_

#include "file.hpp"
#include "core/task/task.hpp"
#include "time_series.hpp"
#include "optimizer/compression_optimizer.hpp"
#include <boost/make_shared.hpp>

namespace ddj
{

class DecompressionTask;
using SharedDecompressionTaskPtr = boost::shared_ptr<DecompressionTask>;

class DecompressionTask : public Task
{
public:
	DecompressionTask(SharedTimeSeriesPtr ts, int columnId)
		: _ts(ts), _columnId(columnId), _deviceId(0)
	{}

	virtual ~DecompressionTask() {}

	DecompressionTask(const DecompressionTask& other)
		: _ts(other._ts),
		  _columnId(other._columnId),
		  _deviceId(other._deviceId)
	{}

public:
	void SetDevice(int deviceId) { _deviceId = deviceId; }

public:
	static SharedDecompressionTaskPtr make_shared(
			SharedTimeSeriesPtr ts,
			int columnId)
	{
		return boost::make_shared<DecompressionTask>(ts, columnId);
	}

protected:
	void execute();

private:
	int _deviceId;
	int _columnId;
	SharedTimeSeriesPtr _ts;
};

} /* namespace ddj */

#endif /* DECOMPRESSION_TASK_HPP_ */
