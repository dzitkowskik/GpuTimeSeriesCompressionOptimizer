/*
 *  not_implemented_exception.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_NOT_IMPLEMENTED_EXCEPTION_HPP_
#define DDJ_NOT_IMPLEMENTED_EXCEPTION_HPP_

namespace ddj {

#include <stdexcept>

class NotImplementedException : public std::logic_error
{
public:
    virtual char const * what() const { return "Function not yet implemented."; }
};

} /* namespace ddj */
#endif /* DDJ_NOT_IMPLEMENTED_EXCEPTION_HPP_ */
