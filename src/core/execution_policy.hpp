/*
 *  execution_policy.hpp
 *
 *  Created on: 15-06-2015
 *      Author: Karol Dzitkowski
 */

#include <cuda_runtime_api.h>

#ifndef DDJ_EXECUTION_POLICY_HPP_
#define DDJ_EXECUTION_POLICY_HPP_

// TODO: Get these from config
#define DEFAULT_CUDA_BLOCK_SIZE 1024
#define MAX_CUDA_BLOCK_DIM_SIZE 1024
#define MAX_CUDA_BLOCK_SIZE 1024
#define MAX_CUDA_GRID_DIM_SIZE 65535

namespace ddj {

class ExecutionPolicy
{
public:
	ExecutionPolicy() : _size(0), _stream(0) {};
	virtual ~ExecutionPolicy(){};

protected:
    size_t _size;
    cudaStream_t _stream;

public:
    void setSize(const size_t size) { this->_size = size; }
    size_t getSize() { return this->_size; }

    virtual dim3 getGrid()
    {
        dim3 grid( (this->_size + DEFAULT_CUDA_BLOCK_SIZE - 1) / DEFAULT_CUDA_BLOCK_SIZE );
        if (grid.x > MAX_CUDA_GRID_DIM_SIZE) {
            grid.y = (grid.x + MAX_CUDA_GRID_DIM_SIZE - 1) / MAX_CUDA_GRID_DIM_SIZE;
            grid.x = MAX_CUDA_GRID_DIM_SIZE;
        }
        return grid;
    }
    virtual dim3 getBlock() { return dim3(DEFAULT_CUDA_BLOCK_SIZE); }
    virtual size_t getShared() { return 0; }

    virtual void setStream(cudaStream_t stream) { _stream = stream; }
    virtual cudaStream_t getStream() { return _stream; }
};

}
#endif /* DDJ_EXECUTION_POLICY_HPP_ */
