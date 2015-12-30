/*
 * compression_task.hpp
 *
 *  Created on: Dec 30, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef COMPRESSION_TASK_HPP_
#define COMPRESSION_TASK_HPP_

#include "core/task/task.hpp"
#include "time_series.hpp"
#include "optimizer/compression_optimizer.hpp"

namespace ddj
{

class CompressionTask : public Task
{
public:
	CompressionTask(SharedTimeSeriesPtr ts, int columnId, SharedCompressionOptimizerPtr optimizer)
		: _ts(ts), _columnId(columnId), _optimizer(optimizer), _deviceId(0)
	{}
	virtual ~CompressionTask() {}

public:
	SharedCudaPtr<char> GetResult() { return _result; }
	void SetDevice(int deviceId) { _deviceId = deviceId; }

protected:
	void execute();

private:
	int _deviceId;
	int _columnId;
	SharedTimeSeriesPtr _ts;
	SharedCompressionOptimizerPtr _optimizer;
	SharedCudaPtr<char> _result;
};

} /* namespace ddj */

#endif /* COMPRESSION_TASK_HPP_ */
