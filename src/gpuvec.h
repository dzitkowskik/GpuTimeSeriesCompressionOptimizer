/*
* gpuvec.h
*
* Created on: Mar 10, 2015
* Author: Karol Dzitkowski
*/

#include "storetypes.h"
#include "config.h"
#include "logger.h"
#include <boost/noncopyable.hpp>
#include <boost/thread/mutex.hpp>

#ifndef DDJ_GPUVEC_H_
#define DDJ_GPUVEC_H_

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
      void* GetBlock(ullint size);

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
