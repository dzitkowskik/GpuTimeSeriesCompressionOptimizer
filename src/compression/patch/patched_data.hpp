/*
 *  patched_data.hpp
 *
 *  Created on: 13-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_PATCHED_DATA_HPP_
#define DDJ_PATCHED_DATA_HPP_

#include "core/cuda_ptr.hpp"
#include "helpers/helper_cudakernels.cuh"
#include "core/execution_policy.hpp"
#include "util/splitter/splitter.hpp"

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
    ExecutionPolicy _policy;
    Splitter _splitter;
};

} /* namespace ddj */
#endif /* DDJ_PATCHED_DATA_HPP_ */
