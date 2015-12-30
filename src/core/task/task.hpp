/*
 * task.hpp
 *
 *  Created on: Dec 30, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_TASK_HPP_
#define DDJ_TASK_HPP_

#include <mutex>
#include <boost/make_shared.hpp>
#include <boost/noncopyable.hpp>

namespace ddj {

enum class TaskStatus
{
	ready,
	executing,
	success,
	failure
};

class Task;
using SharedTaskPtr = boost::shared_ptr<Task>;

class Task : private boost::noncopyable
{
public:
	Task()
		: _status(TaskStatus::ready),
		  _type(""),
		  _id(0)
	{
		_taskDoneMutex.lock();
	}
	virtual ~Task(){}

public:
	void Execute()
	{
		execute();
		_taskDoneMutex.unlock();
	}

	TaskStatus Wait()
	{
		 std::lock_guard<std::mutex> guard(_taskDoneMutex);
		 return _status;
	}

	void Reset()
	{
		_taskDoneMutex.lock();
		_status = TaskStatus::ready;
	}

	// GETTERS AND SETTERS
	std::string GetType() { return _type; }
	void SetType(std::string type) { _type = type; }
	TaskStatus GetStatus() { return _status; }
	void SetStatus(TaskStatus status) { _status = status; }
	int GetId() { return _id; }
	void SetId(int id) { _id = id; }

protected:
	virtual void execute() = 0;

protected:
	std::string _type;
	TaskStatus _status;
	int _id;

private:
	std::mutex _taskDoneMutex;
};

} /* namespace ddj */

#endif /* DDJ_TASK_HPP_ */
