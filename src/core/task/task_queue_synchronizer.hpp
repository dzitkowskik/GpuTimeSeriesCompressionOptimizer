/*
 * task_queue_synchronizer.hpp
 *
 *  Created on: Jan 2, 2016
 *      Author: Karol Dzitkowski
 */

#ifndef TASK_QUEUE_SYNCHRONIZER_HPP_
#define TASK_QUEUE_SYNCHRONIZER_HPP_

#include <boost/thread.hpp>
#include <boost/make_shared.hpp>
#include <queue>

namespace ddj {

class Rutine
{
public:
	virtual void Run() = 0;
};

class TaskQueueSynchronizer
{
public:
	TaskQueueSynchronizer()
	{ _queue = boost::make_shared<std::queue<int>>(); }
	~TaskQueueSynchronizer(){}

public:
	void AddTask(int id) { _queue->push(id); }
	void DoSynchronous(int id, Rutine* r)
	{
		boost::mutex::scoped_lock lock(_mutex);
		while(_queue->front() != id) _cond.wait(lock);
		_queue->pop();
		r->Run();
		_cond.notify_all();
	}

private:
	boost::shared_ptr<std::queue<int>> _queue;
	boost::mutex _mutex;
	boost::condition_variable _cond;
};

} /* namespace ddj */

#endif /* TASK_QUEUE_SYNCHRONIZER_HPP_ */
