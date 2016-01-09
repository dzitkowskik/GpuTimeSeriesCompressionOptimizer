/*
 *  gfc_encoding_impl.cuh
 *
 *  Created on: Nov 17, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef GFC_ENCODING_IMPL_CUH_
#define GFC_ENCODING_IMPL_CUH_

#include "core/cuda_ptr.hpp"

#define ull unsigned long long
#define uchar unsigned char
#define WARPSIZE 32

SharedCudaPtrVector<char> CompressDouble(SharedCudaPtr<double> data, int blocks, int warpsperblock);
SharedCudaPtr<double> DecompressDouble(SharedCudaPtrVector<char> input);

SharedCudaPtrVector<char> CompressFloat(SharedCudaPtr<float> data, int blocks, int warpsperblock);
SharedCudaPtr<float> DecompressFloat(SharedCudaPtrVector<char> input);

#endif /* GFC_ENCODING_IMPL_CUH_ */
