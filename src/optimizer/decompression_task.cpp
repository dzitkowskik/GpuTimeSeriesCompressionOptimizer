/*
 * decompression_task.cpp
 *
 *  Created on: Jan 3, 2016
 *      Author: Karol Dzitkowski
 */

#include "optimizer/decompression_task.hpp"
#include "file.hpp"
#include "core/task/task_queue_synchronizer.hpp"
#include "helpers/helper_device.hpp"

#include <vector>
#include <boost/shared_ptr.hpp>

namespace ddj
{



void DecompressionTask::execute()
{
	// set proper device
	CUDA_CALL( cudaSetDevice(_deviceId) );

	// copy data from ts column to GPU
	auto d_data = CudaPtr<char>::make_shared(_ts->getColumn(_columnId).getSize());
	d_data->fillFromHost(_ts->getColumn(_columnId).getData(), _ts->getColumn(_columnId).getSize());

	// decompress
	auto d_decompressedData = CompressionTree().Decompress(d_data);

	// copy to host as a column of time series
	_ts->getColumn(_columnId).reserveSize(d_decompressedData->size());
	char* h_dataColumn = _ts->getColumn(_columnId).getData();
	CUDA_CALL( cudaMemcpy(h_dataColumn, d_decompressedData->get(), d_decompressedData->size(), CPY_DTH) );

	// end task
	_status = TaskStatus::success;
}

} /* namespace ddj */
