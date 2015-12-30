/*
 * task_scheduler.cpp
 *
 *  Created on: Dec 30, 2015
 *      Author: Karol Dzitkowski
 */

#include "core/task/task_scheduler.hpp"
#include <boost/bind.hpp>

namespace ddj
{

TaskScheduler::TaskScheduler(int threadNumber)
	: _threadNumber(threadNumber)
{
	// Start service
	_work = UniqueWorkPtr(new boost::asio::io_service::work(_ioService));

	// Create threads
	for(int i = 0; i < _threadNumber; i++)
		_threadPool.create_thread(
		    boost::bind(&boost::asio::io_service::run, &_ioService)
		);
}

TaskScheduler::~TaskScheduler()
{
	_ioService.stop();
	_threadPool.join_all();
}

int TaskScheduler::Schedule(SharedTaskPtr task)
{
	int nextId = _taskRegister.size();
	task->SetId(nextId);
	_taskRegister.push_back(task);
	_ioService.post(boost::bind(&Task::Execute, task));
	return nextId;
}

TaskStatus TaskScheduler::Wait(int id)
{
	return _taskRegister[id]->Wait();
}

std::vector<std::pair<int, TaskStatus>> TaskScheduler::WaitAll()
{
	std::vector<std::pair<int, TaskStatus>> result;
	for(auto& task : _taskRegister)
		result.push_back(std::make_pair(task->GetId(), task->Wait()));
	return result;
}

void TaskScheduler::Clear()
{
	_taskRegister.clear();
}

} /* namespace ddj */
