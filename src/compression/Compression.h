/*
 * Compression.h
 *
 *  Created on: 10-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef COMPRESSION_H_
#define COMPRESSION_H_

namespace ddj
{

class CompressionMetadata
{
};

class Compression
{
public:
	Compression();
	virtual ~Compression();

	/* NOT A TEMPLATE FOR A REASON */
	virtual void* Encode(int* data, int in_size, int& out_size, CompressionMetadata& metadata) = 0;
	virtual void* Decode(int* data, int in_size, int& out_size, CompressionMetadata metadata) = 0;

	virtual void* Encode(unsigned int* data, int in_size, int& out_size, CompressionMetadata& metadata) = 0;
	virtual void* Decode(unsigned int* data, int in_size, int& out_size, CompressionMetadata metadata) = 0;

	virtual void* Encode(long int* data, int in_size, int& out_size, CompressionMetadata& metadata) = 0;
	virtual void* Decode(long int* data, int in_size, int& out_size, CompressionMetadata metadata) = 0;

	virtual void* Encode(unsigned long int* data, int in_size, int& out_size, CompressionMetadata& metadata) = 0;
	virtual void* Decode(unsigned long int* data, int in_size, int& out_size, CompressionMetadata metadata) = 0;

	virtual void* Encode(long long int* data, int in_size, int& out_size, CompressionMetadata& metadata) = 0;
	virtual void* Decode(long long int* data, int in_size, int& out_size, CompressionMetadata metadata) = 0;

	virtual void* Encode(unsigned long long int* data, int in_size, int& out_size, CompressionMetadata& metadata) = 0;
	virtual void* Decode(unsigned long long int* data, int in_size, int& out_size, CompressionMetadata metadata) = 0;

	virtual void* Encode(float* data, int in_size, int& out_size, CompressionMetadata& metadata) = 0;
	virtual void* Decode(float* data, int in_size, int& out_size, CompressionMetadata metadata) = 0;

	virtual void* Encode(double* data, int in_size, int& out_size, CompressionMetadata& metadata) = 0;
	virtual void* Decode(double* data, int in_size, int& out_size, CompressionMetadata metadata) = 0;
};

} /* namespace ddj */
#endif /* COMPRESSION_H_ */
