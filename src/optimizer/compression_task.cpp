/*
 * compression_task.cpp
 *
 *  Created on: Dec 30, 2015
 *      Author: Karol Dzitkowski
 */

#include "helpers/helper_device.hpp"
#include "optimizer/compression_task.hpp"

namespace ddj
{

void CompressionTask::execute()
{
	// set proper device
	CUDA_CALL( cudaSetDevice(_deviceId) );

	// copy data to device
	auto h_data = _ts->getColumn(_columnId).getData();
	auto size = _ts->getColumn(_columnId).getSize();
	auto d_data = CudaPtr<char>::make_shared(size);
	d_data->fillFromHost(h_data, size);

	// compress data
	_result = _optimizer->CompressData(d_data, _ts->getColumn(_columnId).getType());
}

} /* namespace ddj */
