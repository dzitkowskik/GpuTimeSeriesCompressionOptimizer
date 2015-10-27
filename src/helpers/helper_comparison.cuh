/*
 * helper_comparison.cuh 26-03-2015 Karol Dzitkowski
 */

#ifndef DDJ_HELPER_COMPARISON_H_
#define DDJ_HELPER_COMPARISON_H_

#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define COMP_THREADS_PER_BLOCK 512

template <typename T> bool CompareDeviceArrays(T* a, T* b, int size);

// TODO: Dopisać funkcje przyjmujące SharedCudaPtr

#endif
