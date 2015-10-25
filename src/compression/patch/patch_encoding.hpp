/*
 *  patched_data.hpp
 *
 *  Created on: 13-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_PATCHED_DATA_HPP_
#define DDJ_PATCHED_DATA_HPP_

#include "core/cuda_ptr.hpp"
#include "core/execution_policy.hpp"
#include "util/splitter/splitter.hpp"
#include "compression/encoding.hpp"

#include <tuple>
#include <boost/noncopyable.hpp>

namespace ddj {

template<typename UnaryOperator>
class PatchEncoding : public Encoding
{
public:
    PatchEncoding(UnaryOperator op) : _op(op) {};
	virtual ~PatchEncoding() {};
	PatchEncoding(const PatchEncoding& other) : _op(other._op) {}
	PatchEncoding(PatchEncoding&& other) : _op(std::move(other._op)) {}

public:
	unsigned int GetNumberOfResults() { return 2; }

protected:
	SharedCudaPtrVector<char> EncodeInt(SharedCudaPtr<int> data);
	SharedCudaPtr<int> DecodeInt(SharedCudaPtrVector<char> data);
	SharedCudaPtrVector<char> EncodeFloat(SharedCudaPtr<float> data);
	SharedCudaPtr<float> DecodeFloat(SharedCudaPtrVector<char> data);

// TODO: Should be also private
public:
	template<typename T> SharedCudaPtrVector<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtrVector<char> data);

private:
	UnaryOperator _op;
    ExecutionPolicy _policy;
    Splitter _splitter;
};

} /* namespace ddj */
#endif /* DDJ_PATCHED_DATA_HPP_ */
