#include "gpuvec.h"
#include <cuda_runtime_api.h>

namespace ddj
{

    GpuVec::GpuVec()
    {
    }

    GpuVec::~GpuVec()
    {
    }

    ullint Write(void* data, ullint size)
    {
        return 0;
    }

    void* Read(ullint offset, ullint size)
    {
        return nullptr;
    }

    ullint Size()
    {
        return 0;
    }

    ullint GetMemoryOffset()
    {
        return 0;
    }

    void SetMemoryOffset(ullint offset)
    {
    }

    void* GetFirstFreeAddress()
    {
        return nullptr;
    }

    ullint GetCapacity()
    {
        return 0;
    }

    void Resize(ullint size)
    {
    }

} /* namespace ddj */
