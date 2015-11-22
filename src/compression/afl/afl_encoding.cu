#include "compression/afl/afl_encoding.hpp"
#include "afl_gpu.cuh"
#include "util/statistics/cuda_array_statistics.hpp"
#include "util/transform/cuda_array_transform.hpp"
#include "util/copy/cuda_array_copy.hpp"
#include "util/stencil/stencil.hpp"
#include "helpers/helper_float.hpp"
#include "core/cuda_launcher.cuh"
#include "helpers/helper_print.hpp"

namespace ddj
{

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
	auto metadata = CudaPtr<char>::make_shared(4*sizeof(char));

	char* host_metadata;
	CUDA_CALL( cudaMallocHost(&host_metadata, 4) );
	host_metadata[0] = minBit;
	host_metadata[1] = rest;

	run_afl_compress_gpu<int, 1>(
		minBit, data->get(), (int*)result->get(), data->size(), comprDataSize/sizeof(int));

	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		printf("cuda error\n");
		printf("post-kernel err is %s.\n", cudaGetErrorString(err));
	    exit(1);
	}

	metadata->fillFromHost(host_metadata, 4*sizeof(char));
	CUDA_CALL( cudaFreeHost(host_metadata) );

	cudaDeviceSynchronize();
	return SharedCudaPtrVector<char> {metadata, result};
}

__global__ void _splitFloatKernel(
		float* data,
		size_t size,
		int* mantissa,
		int* exponent,
		int* sign)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	floatCastUnion fu { data[idx] };

	mantissa[idx] = fu.parts.mantisa;
	exponent[idx] = fu.parts.exponent;
	sign[idx] = fu.parts.sign;
}

template<>
SharedCudaPtr<int> AflEncoding::Decode(SharedCudaPtrVector<char> input);

template<>
SharedCudaPtrVector<char> AflEncoding::Encode(SharedCudaPtr<float> data)
{
	auto minMax = CudaArrayStatistics().MinMax(data);
	char allPositive = std::get<0>(minMax) >= 0 ? 1 : 0;
	char allNegative = std::get<1>(minMax) < 0 ? 2 : 0;
	char sign = allPositive + allNegative;
	auto signResult = CudaPtr<int>::make_shared(data->size());
	auto exponentResult = CudaPtr<int>::make_shared(data->size());
	auto mantissaResult = CudaPtr<int>::make_shared(data->size());

	// Now we split every float number to three integers - sign, exponent and mantissa
	this->_policy.setSize(data->size());
	cudaLaunch(this->_policy, _splitFloatKernel,
			data->get(),
			data->size(),
			mantissaResult->get(),
			exponentResult->get(),
			signResult->get());
	cudaDeviceSynchronize();

	// We do AFL encoding on mantissa and exponent int arrays
	auto resultVector = Encode(mantissaResult);
	auto resultVector2 = Encode(exponentResult);

	resultVector.insert(resultVector.end(), resultVector2.begin(), resultVector2.end());

	// Save the size of mantissa after compression
	// When all numbers are positive or negative we save sign only in metadata as one char
	// Else we save a stencil containing which numbers are negative
	SharedCudaPtr<char> metadata;
	metadata = CudaPtr<char>::make_shared(sizeof(size_t) + 1);
	size_t size = resultVector[1]->size();

	CUDA_CALL( cudaMemcpy(metadata->get(), &size, sizeof(size_t), CPY_HTD) );
	CUDA_CALL( cudaMemcpy(metadata->get()+sizeof(size_t), &sign, 1, CPY_HTD) );
	if(sign == 0)
	{
		auto stencil = Stencil(signResult).pack();
		metadata = CudaArrayCopy().Concatenate(SharedCudaPtrVector<char>{metadata, stencil});
	}

	return SharedCudaPtrVector<char>{ metadata, CudaArrayCopy().Concatenate(resultVector) };
}

template<typename T>
SharedCudaPtr<T> DecodeAfl(T* data, size_t size, int minBit, int rest)
{
	// Calculate length
	int comprBits = size * 8 - rest;
	int length = comprBits / minBit;

	auto result = CudaPtr<int>::make_shared(length);
	run_afl_decompress_gpu<int, 1>(minBit, data, result->get(), length);
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		printf("cuda error\n");
		printf("post-kernel err is %s.\n", cudaGetErrorString(err));
		exit(1);
	}
	return result;
}

template<>
SharedCudaPtr<int> AflEncoding::Decode(SharedCudaPtrVector<char> input)
{
	auto metadata = input[0]->copyToHost();
	auto data = input[1];

	// Get min bit and rest
	int minBit = (*metadata)[0];
	int rest = (*metadata)[1];

	return DecodeAfl<int>((int*)data->get(), data->size(), minBit, rest);
}

__global__ void _composeFloatKernel(
		int* mantissa,
		int* exponent,
		int* sign,
		size_t size,
		float* result)
{
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	floatCastUnion fu;
	fu.parts.sign = sign[idx];
	fu.parts.exponent = exponent[idx];
	fu.parts.mantisa = mantissa[idx];
	result[idx] = fu.value;
}

template<>
SharedCudaPtr<float> AflEncoding::Decode(SharedCudaPtrVector<char> input)
{
	int offset = 0, step = sizeof(char);

	auto metadata = input[0];
	auto data = input[1];

	// read metadata information
	char sign;
	long int compressedMantissaSize;
	CUDA_CALL( cudaMemcpy(&compressedMantissaSize, metadata->get(), sizeof(size_t), CPY_DTH) );
	CUDA_CALL( cudaMemcpy(&sign, metadata->get()+sizeof(size_t), 1, CPY_DTH) );

	// read mantissa metadata information
	char minBit, rest;
	CUDA_CALL( cudaMemcpy(&minBit, data->get()+offset, step, CPY_DTH) );
	offset += step;
	CUDA_CALL( cudaMemcpy(&rest, data->get()+offset, step, CPY_DTH) );
	offset += 3*step;

	// decode mantissa
	auto mantissaDecoded = DecodeAfl<int>((int*)(data->get()+offset), compressedMantissaSize, minBit, rest);

	long int compressedExponentSize = data->size() - compressedMantissaSize - 8;
	offset += compressedMantissaSize;

	// read exponent metadata information
	CUDA_CALL( cudaMemcpy(&minBit, data->get()+offset, step, CPY_DTH) );
	offset += step;
	CUDA_CALL( cudaMemcpy(&rest, data->get()+offset, step, CPY_DTH) );
	offset += 3*step;

	// decode exponent
	auto exponentDecoded = DecodeAfl<int>((int*)(data->get()+offset), compressedExponentSize, minBit, rest);

	// recover signs
	Stencil stencil;
	size_t size = mantissaDecoded->size();
	if(sign)
		stencil = Stencil(
				CudaArrayTransform().Transform<int, int>(
						CudaPtr<int>::make_shared(size),
						FillOperator<int, int> {(int)sign-1}));
	else
		stencil = Stencil(metadata, sizeof(size_t)+1);

	// compose exponent, mantissa and sign to floats
	auto result = CudaPtr<float>::make_shared(size);
	this->_policy.setSize(size);
	cudaLaunch(this->_policy, _composeFloatKernel,
			mantissaDecoded->get(),
			exponentDecoded->get(),
			stencil->get(),
			size,
			result->get());
	cudaDeviceSynchronize();

	return result;
}

SharedCudaPtrVector<char> AflEncoding::EncodeInt(SharedCudaPtr<int> data)
{
	if(data->size() <= 0)
		return SharedCudaPtrVector<char>{ CudaPtr<char>::make_shared(), CudaPtr<char>::make_shared() };

	return this->Encode<int>(data);
}

SharedCudaPtr<int> AflEncoding::DecodeInt(SharedCudaPtrVector<char> data)
{
	if(data[1]->size() <= 0)
		return CudaPtr<int>::make_shared();

	return this->Decode<int>(data);
}

SharedCudaPtrVector<char> AflEncoding::EncodeFloat(SharedCudaPtr<float> data)
{
	if(data->size() <= 0)
		return SharedCudaPtrVector<char>{ CudaPtr<char>::make_shared(), CudaPtr<char>::make_shared() };

	return this->Encode<float>(data);
}

SharedCudaPtr<float> AflEncoding::DecodeFloat(SharedCudaPtrVector<char> data)
{
	if(data[1]->size() <= 0)
		return CudaPtr<float>::make_shared();

	return this->Decode<float>(data);
}

SharedCudaPtrVector<char> AflEncoding::EncodeDouble(SharedCudaPtr<double> data)
{ return SharedCudaPtrVector<char>(); }
SharedCudaPtr<double> AflEncoding::DecodeDouble(SharedCudaPtrVector<char> data)
{ return SharedCudaPtr<double>(); }

size_t AflEncoding::GetMetadataSize(SharedCudaPtr<char> data, DataType type)
{
	if(data->size() <= 0) return 0;
	switch(type)
	{
		case DataType::d_int:
			return 4*sizeof(char);
		case DataType::d_float:
			return sizeof(size_t) + 1;
		default:
			throw NotImplementedException("No DictEncoding::GetCompressedSize implementation for that type");
	}
}

size_t AflEncoding::GetCompressedSize(SharedCudaPtr<char> data, DataType type)
{
	if(data->size() <= 0) return 0;
	switch(type)
	{
		case DataType::d_int:
			return GetCompressedSizeIntegral(CastSharedCudaPtr<char, int>(data));
		case DataType::d_float:
			return GetCompressedSizeFloatingPoint(CastSharedCudaPtr<char, float>(data));
		default:
			throw NotImplementedException("No DictEncoding::GetCompressedSize implementation for that type");
	}
}

template<typename T>
size_t AflEncoding::GetCompressedSizeIntegral(SharedCudaPtr<T> data)
{
	char minBit = CudaArrayStatistics().MinBitCnt<int>(data);

	int elemBitSize = 8*sizeof(int);
	int comprElemCnt = (minBit * data->size() + elemBitSize - 1) / elemBitSize;
	int comprDataSize = comprElemCnt * sizeof(int);

	return comprDataSize;
}

template<typename T>
size_t AflEncoding::GetCompressedSizeFloatingPoint(SharedCudaPtr<T> data)
{
	auto minMax = CudaArrayStatistics().MinMax(data);
	auto signResult = CudaPtr<int>::make_shared(data->size());
	auto exponentResult = CudaPtr<int>::make_shared(data->size());
	auto mantissaResult = CudaPtr<int>::make_shared(data->size());

	// Now we split every float number to three integers - sign, exponent and mantissa
	this->_policy.setSize(data->size());
	cudaLaunch(this->_policy, _splitFloatKernel,
			data->get(),
			data->size(),
			mantissaResult->get(),
			exponentResult->get(),
			signResult->get());
	cudaDeviceSynchronize();

	size_t size = GetCompressedSizeIntegral(exponentResult) + GetCompressedSizeIntegral(mantissaResult);
	size += GetMetadataSize(CastSharedCudaPtr<int, char>(exponentResult), DataType::d_int);
	size += GetMetadataSize(CastSharedCudaPtr<int, char>(mantissaResult), DataType::d_int);
	return size;
}

} /* namespace ddj */
