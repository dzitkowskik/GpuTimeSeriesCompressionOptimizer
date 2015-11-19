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
#include "util/statistics/cuda_array_statistics.hpp"
#include "compression/data_type.hpp"
#include "compression/encoding_type.hpp"
#include "compression/patch/patch_type.hpp"
#include "compression/encoding.hpp"

namespace ddj {

class PatchEncoding : public Encoding
{
public:
	PatchEncoding(PatchType type) : _type(type)	{}
	virtual ~PatchEncoding(){}
	PatchEncoding(const PatchEncoding& other) : _type(other._type) {}
	PatchEncoding(PatchEncoding&& other) : _type(std::move(other._type)) {}

public:
	unsigned int GetNumberOfResults() { return 2; }

	size_t GetMetadataSize(SharedCudaPtr<char> data, DataType type)
	{
		if(data->size() <= 0) return 0;
		int elemCnt = data->size() / GetDataTypeSize(type);
		return (elemCnt + 7) / 8 + 1;
	}

	size_t GetCompressedSize(SharedCudaPtr<char> data, DataType type)
	{
		return data->size();
	}

protected:
	SharedCudaPtrVector<char> EncodeInt(SharedCudaPtr<int> data) = 0;
	SharedCudaPtr<int> DecodeInt(SharedCudaPtrVector<char> data) = 0;
	SharedCudaPtrVector<char> EncodeFloat(SharedCudaPtr<float> data) = 0;
	SharedCudaPtr<float> DecodeFloat(SharedCudaPtrVector<char> data) = 0;
	SharedCudaPtrVector<char> EncodeDouble(SharedCudaPtr<double> data) = 0;
	SharedCudaPtr<double> DecodeDouble(SharedCudaPtrVector<char> data) = 0;

protected:
    ExecutionPolicy _policy;
    Splitter _splitter;
    PatchType _type;
};

} /* namespace ddj */
#endif /* DDJ_PATCHED_DATA_HPP_ */
