/*
* gpuvec.h
*
* Created on: Mar 10, 2015
* Author: Karol Dzitkowski
*/

#ifndef DDJ_GPUVEC_H_
#define DDJ_GPUVEC_H_

#include "storetypes.h"
#include "config.h"
#include "logger.h"
#include <boost/noncopyable.hpp>
#include <boost/thread/mutex.hpp>

namespace ddj {

class GpuVec : public boost::noncopyable
{
public:
    GpuVec();
    virtual ~GpuVec();

private:
    /* STORE MEMORY (on GPU) */
    ullint _memoryOffset;
    boost::mutex _offsetMutex;
    void* _memoryPointer;
    ullint _memoryCapacity;

    Logger _logger = Logger::getRoot();
    Config* _config = Config::GetInstance();

public:
    ullint Write(void* data, ullint size);
    void* Read(ullint offset, ullint size);
    ullint Size();

private:
    ullint GetMemoryOffset();
    void SetMemoryOffset(ullint offset);
    void* GetFirstFreeAddress();
    ullint GetCapacity();
    void Resize(ullint size);

private:
    void allocateGpuStorage();
};

} /* namespace ddj */
#endif /* DDJ_GPUVEC_H_ */
