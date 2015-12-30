/*
 * task_scheduler.hpp
 *
 *  Created on: Dec 30, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_TASK_SCHEDULER_HPP_
#define DDJ_TASK_SCHEDULER_HPP_

#include "task.hpp"

#include <boost/asio/io_service.hpp>
#include <boost/thread/thread.hpp>
#include <boost/noncopyable.hpp>
#include <vector>
#include <queue>
#include <memory>

namespace ddj {

class TaskScheduler;
using UniqueTaskSchedulerPtr = std::unique_ptr<TaskScheduler>;
using UniqueWorkPtr = std::unique_ptr<boost::asio::io_service::work>;

class TaskScheduler : private boost::noncopyable
{
public:
	virtual ~TaskScheduler();

public:
	int Schedule(SharedTaskPtr task);
	TaskStatus Wait(int id);
	std::vector<std::pair<int, TaskStatus>> WaitAll();
	void Clear();

public:
	static UniqueTaskSchedulerPtr make_unique(int threadNumber)
	{
		return std::unique_ptr<TaskScheduler>(new TaskScheduler(threadNumber));
	}

private:
	TaskScheduler(int threadNumber);

private:
	int _threadNumber;
	UniqueWorkPtr _work;
	boost::asio::io_service _ioService;
	boost::thread_group _threadPool;
	std::vector<SharedTaskPtr> _taskRegister;
};

} /* namespace ddj */

#endif /* DDJ_TASK_SCHEDULER_HPP_ */
