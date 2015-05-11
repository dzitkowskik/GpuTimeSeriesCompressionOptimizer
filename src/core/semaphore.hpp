/*
 * semaphore.hpp
 *
 *  Created on: 30-10-2013
 *      Author: Karol Dzitkowski
 */

#ifndef SEMAPHORE_H_
#define SEMAPHORE_H_

#include <boost/thread.hpp>

namespace ddj
{

class Semaphore
{
private:
	unsigned int _max;
	unsigned int _value;
	boost::mutex _mutex;
	boost::condition_variable _cond;

public:
	Semaphore(unsigned int max);
	virtual ~Semaphore();

public:
	void Wait();
	void Release();
};

} /* namespace ddj */
#endif /* SEMAPHORE_H_ */
