/*
 * compression_task.cpp
 *
 *  Created on: Dec 30, 2015
 *      Author: Karol Dzitkowski
 */

#include "data/file.hpp"
#include "core/task/task_queue_synchronizer.hpp"
#include "core/cuda_device.hpp"
#include "optimizer/compression_task.hpp"

#include <vector>
#include <boost/shared_ptr.hpp>

namespace ddj
{

class FileWritterRutine : public Rutine
{
public:
	FileWritterRutine(boost::shared_ptr<std::vector<char>> data, File& destination)
		: _data(data), _destination(destination)
	{}
	virtual ~FileWritterRutine(){}

public:
	void Run();

private:
	boost::shared_ptr<std::vector<char>> _data;
	File _destination;
};

void FileWritterRutine::Run()
{
	size_t size = _data->size();
	_destination.WriteRaw((char*)&size, sizeof(size_t));
	_destination.WriteRaw(_data->data(), size);
}

void CompressionTask::execute()
{
	// set proper device
	CUDA_CALL( cudaSetDevice(_deviceId) );
	LOG4CPLUS_TRACE_FMT(_logger, "Compression task %d started on device %d", _id, _deviceId);

	// copy data to device
	auto h_data = _ts->getColumn(_columnId).getData();
	auto size = _ts->getColumn(_columnId).getSize();
	auto d_data = CudaPtr<char>::make_shared(size);
	d_data->fillFromHost(h_data, size);

	// compress data
	auto type = _ts->getColumn(_columnId).getType();
	LOG4CPLUS_DEBUG_FMT(_logger, "Task id = %d, compress type %s with size %lu",
		_id, GetDataTypeString(type).c_str(), d_data->size());
	////////////////////////////////////////////////////////////////////
	auto d_result = _optimizer->CompressData(d_data, type);
	////////////////////////////////////////////////////////////////////
	LOG4CPLUS_DEBUG_FMT(_logger, "Task id = %d, compressed to size %lu",
		_id, d_result->size());

	// check for errors
	CUDA_CALL( cudaGetLastError() );

	// send compressed batch to host
	auto h_result = d_result->copyToHost();

	// synchronously write data to stream in order of scheduled tasks
	FileWritterRutine rutine(h_result, _outputFile);
	_synchronizer->DoSynchronous(_id, &rutine);

	// end task
	_status = TaskStatus::success;
	LOG4CPLUS_TRACE_FMT(_logger, "Compression task %d ended on device %d", _id, _deviceId);
}

} /* namespace ddj */
