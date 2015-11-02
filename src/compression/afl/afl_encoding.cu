#include "compression/afl/afl_encoding.hpp"
#include "compression/afl/afl_encoding_impl.cuh"
#include "util/statistics/cuda_array_statistics.hpp"

namespace ddj
{

size_t AflEncoding::GetCompressedSize(SharedCudaPtr<char> data, DataType type)
{
	SharedCudaPtr<int> intData;
	SharedCudaPtr<float> floatData;
	char minBit;
	int size;

	switch(type)
	{
		case DataType::d_int:
			intData = boost::reinterpret_pointer_cast<CudaPtr<int>>(data);
			minBit = CudaArrayStatistics().MinBitCnt<int>(intData);
			size = intData->size();
			break;
		case DataType::d_float:
			floatData = boost::reinterpret_pointer_cast<CudaPtr<float>>(data);
			minBit = CudaArrayStatistics().MinBitCnt<float>(floatData);
			size = floatData->size();
			break;
		default:
			throw NotImplementedException("No getCompressedType method for that type!");
	}

	int elemBitSize = 8*sizeof(int);
	int comprElemCnt = (minBit * size + elemBitSize - 1) / elemBitSize;
	int comprDataSize = comprElemCnt * sizeof(int);
	return comprDataSize;
}

template<>
SharedCudaPtrVector<char> AflEncoding::Encode(SharedCudaPtr<int> data)
{
	// Get minimal bit count needed to encode data
	char minBit = CudaArrayStatistics().MinBitCnt<int>(data);
	int elemBitSize = 8*sizeof(int);
	int comprElemCnt = (minBit * data->size() + elemBitSize - 1) / elemBitSize;
	int comprDataSize = comprElemCnt * sizeof(int);
	char rest = (comprDataSize*8) - (data->size()*minBit);

	auto result = CudaPtr<char>::make_shared(comprDataSize);
	auto metadata = CudaPtr<char>::make_shared(2*sizeof(char));

	char host_metadata[2];
	host_metadata[0] = minBit;
	host_metadata[1] = rest;

	run_afl_compress_gpu<int, 32>(
		minBit, data->get(), (int*)result->get(), data->size());

	metadata->fillFromHost(host_metadata, 2*sizeof(char));

	cudaDeviceSynchronize();
	return SharedCudaPtrVector<char> {metadata, result};
}

template<>
SharedCudaPtr<int> AflEncoding::Decode(SharedCudaPtrVector<char> input)
{
	auto metadata = input[0]->copyToHost();
	auto data = input[1];

	// Get min bit and rest
	int minBit = (*metadata)[0];
	int rest = (*metadata)[1];

	// Calculate length
	int comprElemCnt = data->size()/sizeof(int);
	int comprBits = data->size() * 8 - rest;
	int length = comprBits / minBit;

	auto result = CudaPtr<int>::make_shared(length);
	run_afl_decompress_gpu<int, 32>(minBit, (int*)data->get(), result->get(), length);

	cudaDeviceSynchronize();
	return result;
}

SharedCudaPtrVector<char> AflEncoding::EncodeInt(SharedCudaPtr<int> data)
{
	return this->Encode<int>(data);
}

SharedCudaPtr<int> AflEncoding::DecodeInt(SharedCudaPtrVector<char> data)
{
	return this->Decode<int>(data);
}

SharedCudaPtrVector<char> AflEncoding::EncodeFloat(SharedCudaPtr<float> data)
{
	throw NotImplementedException("Afl encoding for this data type is not implemented");
//		return this->Encode<float>(data);
}

SharedCudaPtr<float> AflEncoding::DecodeFloat(SharedCudaPtrVector<char> data)
{
	throw NotImplementedException("Afl decoding for this data type is not implemented");
//		return this->Decode<float>(data);
}

} /* namespace ddj */
