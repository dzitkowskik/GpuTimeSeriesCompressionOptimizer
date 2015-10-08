/*
 *  stencil.hpp
 *
 *  Created on: 09-08-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_UTIL_STENCIL_HPP_
#define DDJ_UTIL_STENCIL_HPP_

#include "core/cuda_ptr.hpp"
#include "core/execution_policy.hpp"

namespace ddj {

class Stencil
{
public:
	Stencil() {}
	Stencil(SharedCudaPtr<char> data);
    Stencil(SharedCudaPtr<int> data) { _data = data; };
    Stencil(const Stencil& other) : _data(other._data) {}
    Stencil(Stencil&& other) noexcept : _data(std::move(other._data)) {}
    ~Stencil() {}

    SharedCudaPtr<int> operator->() const
    { return this->_data; }

    SharedCudaPtr<int> operator*() const
    { return this->_data; }

    SharedCudaPtr<char> pack();
    SharedCudaPtr<int> unpack(SharedCudaPtr<char> data);

private:
    SharedCudaPtr<int> _data;
    ExecutionPolicy _policy;
};

} /* namespace ddj */
#endif /* DDJ_UTIL_STENCIL_HPP_ */