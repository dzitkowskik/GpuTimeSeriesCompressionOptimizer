#include "util/histogram/histogram.hpp"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

namespace ddj {

template<typename T>
SharedCudaPtr<T> Histogram::GetMostFrequentSparse(SharedCudaPtrPair<T, int> histogram, int freqCnt)
{
	thrust::device_ptr<T> keys_ptr(histogram.first->get());
	thrust::device_ptr<int> counts_ptr(histogram.second->get());
	int N = histogram.first->size();

	// sort to have greater counts first
	thrust::sort_by_key(counts_ptr, counts_ptr + N, keys_ptr, thrust::greater<int>());

	// get first freqCnt keys
	auto result = CudaPtr<T>::make_shared(freqCnt);
	thrust::device_ptr<T> result_ptr(result->get());
	thrust::copy_n(keys_ptr, freqCnt, result_ptr);

	return result;
}

#define MOST_FREQ_SPARSE_SPEC(X) \
	template SharedCudaPtr<X> Histogram::GetMostFrequentSparse<X>(SharedCudaPtrPair<X, int>, int);
FOR_EACH(MOST_FREQ_SPARSE_SPEC, float, double, int, unsigned int, long int, long long int)

} /* namespace ddj */
