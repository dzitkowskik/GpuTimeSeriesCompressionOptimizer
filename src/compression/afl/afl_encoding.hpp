#ifndef DDJ_AFL_ENCODING_HPP_
#define DDJ_AFL_ENCODING_HPP_

#include <boost/noncopyable.hpp>
#include "core/cuda_ptr.hpp"
#include "helpers/helper_cudakernels.cuh"

namespace ddj
{

class AflEncoding
{
private:
	HelperCudaKernels _cudaKernels;

public:
	template<typename T> SharedCudaPtr<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtr<char> data);

private:
    template<typename T> int getMinBitCnt(SharedCudaPtr<T> data);
};

} /* namespace ddj */
#endif /* DDJ_AFL_ENCODING_HPP_ */
