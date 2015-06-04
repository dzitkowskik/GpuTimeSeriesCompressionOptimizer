/*
 *  simple_patch.cuh
 *
 *  Created on: 13-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_SPLITTING_SIMPLE_PATCH_CUH_
#define DDJ_SPLITTING_SIMPLE_PATCH_CUH_

#include "core/cuda_ptr.hpp"
#include "helpers/helper_cudakernels.cuh"
#include <tuple>
#include <boost/noncopyable.hpp>

namespace ddj {

template<typename DataType, typename UnaryOperator>
class PatchedData : private boost::noncopyable
{
    typedef std::tuple<SharedCudaPtr<DataType>, SharedCudaPtr<DataType>> PartData;
    typedef SharedCudaPtr<int> Stencil;

public:
    PatchedData(UnaryOperator op);
	virtual ~PatchedData();

public:
	void Init(SharedCudaPtr<DataType> data);
	void Init(SharedCudaPtr<char> data);

	SharedCudaPtr<DataType> GetFirst();
	SharedCudaPtr<DataType> GetSecond();

	SharedCudaPtr<char> Encode();
	SharedCudaPtr<DataType> Decode();

private:
	PartData _data;
	Stencil _stencil;
	UnaryOperator _op;
    HelperCudaKernels _kernels;
};

} /* namespace ddj */
#endif /* DDJ_SPLITTING_SIMPLE_PATCH_CUH_ */
