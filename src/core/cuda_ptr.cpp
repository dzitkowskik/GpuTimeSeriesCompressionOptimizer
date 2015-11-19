#include "cuda_ptr.hpp"
#include "macros.h"

template<typename T>
SharedCudaPtr<T> Concatenate(SharedCudaPtrVector<T> data)
{
	size_t totalSize = 0, i = 0, offset = 0;
	cudaStream_t* streams = new cudaStream_t[data.size()];
	// get total data parts count and prepare so many streams
	for(auto& part : data)
	{
		totalSize += part->size();
		CUDA_CALL( cudaStreamCreate(&streams[i++]) );
	}
	auto result = CudaPtr<T>::make_shared(totalSize);

	// copy each part to output with offset in different stream
	i = 0;
	for(auto& part : data)
	{
		if(part->size() > 0)
		{
			CUDA_CALL(
				cudaMemcpyAsync(
					result->get()+offset,
					part->get(),
					part->size()*sizeof(T),
					CPY_DTD,
					streams[i++])
			);
			offset += part->size();
		}
	}

	// synchronize and destroy streams
	for(i=0; i < data.size(); i++)
		CUDA_CALL( cudaStreamSynchronize(streams[i]) );
	for(i=0; i < data.size(); i++)
		CUDA_CALL( cudaStreamDestroy(streams[i]) );
	cudaDeviceSynchronize();
	return result;
}

#define CUDA_PTR_SPEC(X) \
	template SharedCudaPtr<X> Concatenate<X>(SharedCudaPtrVector<X>);
FOR_EACH(CUDA_PTR_SPEC, char, double, float, int, long, long long, unsigned int)
