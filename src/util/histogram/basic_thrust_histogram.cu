#include "basic_thrust_histogram.cuh"

namespace ddj {

SharedCudaPtrPair<int, int> BasicThrustHistogram::IntegerHistogram(SharedCudaPtr<int> data)
{
    return SharedCudaPtrPair<int, int>(SharedCudaPtr<int>(), SharedCudaPtr<int>());
}

} /* namespace ddj */
