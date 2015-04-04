/*
* gpuvec.h
*
* Created on: Mar 10, 2015
* Author: Karol Dzitkowski
*/

#ifndef DDJ_GPUVEC_H_
#define DDJ_GPUVEC_H_

#include "storetypes.h"
#include "../core/config.h"
#include "../core/logger.h"
#include <boost/noncopyable.hpp>
#include <boost/thread/mutex.hpp>

namespace ddj
{

class GpuVec : public boost::noncopyable
{
private:
    /* STORE MEMORY (on GPU) */
    ullint _memoryOffset;
    boost::mutex _offsetMutex;
    void* _memoryPointer;
    ullint _memoryCapacity;

private:
    /* CONFIG & LOGGER */
    Logger _logger = Logger::getRoot();
    Config* _config = Config::GetInstance();

public:
    GpuVec();
    GpuVec(ullint size);
    virtual ~GpuVec();

public:
    ullint Write(void* data, ullint size);
    void* Read(ullint offset, ullint size);
    void* Get(ullint offset, ullint size);
    ullint Size();

private:
    void* GetFirstFreeAddress();
    ullint GetCapacity();
    void Resize(ullint size);
    void Init();
};

} /* namespace ddj */
#endif /* DDJ_GPUVEC_H_ */
