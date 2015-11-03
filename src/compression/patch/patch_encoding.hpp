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
#include "core/not_implemented_exception.hpp"
#include "util/stencil/stencil_operators.hpp"
#include "util/splitter/splitter.hpp"
#include "compression/data_type.hpp"
#include "compression/encoding_type.hpp"
#include "compression/patch/patch_type.hpp"
#include "compression/encoding.hpp"
#include "compression/encoding_factory.hpp"

#include <tuple>
#include <boost/noncopyable.hpp>
#include <boost/make_shared.hpp>

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

template<typename T>
class PatchEncodingFactory : public EncodingFactory
{
public:
	PatchType patchType;

	T min;
	T max;
	T factor;

	PatchEncodingFactory(DataType dt, PatchType pt)
		: EncodingFactory(dt, EncodingType::patch), patchType(pt)
	{}
	~PatchEncodingFactory(){}
	PatchEncodingFactory(const PatchEncodingFactory& other)
		: EncodingFactory(other.dataType, EncodingType::patch), patchType(other.patchType)
	{}

	boost::shared_ptr<Encoding> Get()
	{
		OutsideOperator<T> op;
		return boost::make_shared<PatchEncoding<OutsideOperator<T>>>(op);
	}

	boost::shared_ptr<Encoding> Get(SharedCudaPtr<char> data)
	{
		OutsideOperator<T> op;
		T dist = max - min;

		switch(patchType)
		{
			case PatchType::outside:
				op.low = min + factor * dist;
				op.high = max - factor * dist;
				return boost::make_shared<PatchEncoding<OutsideOperator<T>>>(op);
			default:
				throw NotImplementedException("Encoding of this type not implemented");
		}
	}
};

} /* namespace ddj */
#endif /* DDJ_PATCHED_DATA_HPP_ */
