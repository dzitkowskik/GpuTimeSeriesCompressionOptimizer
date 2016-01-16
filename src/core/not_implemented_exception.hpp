/*
 *  not_implemented_exception.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_NOT_IMPLEMENTED_EXCEPTION_HPP_
#define DDJ_NOT_IMPLEMENTED_EXCEPTION_HPP_

#include <stdexcept>
#include <string>

namespace ddj {

class NotImplementedException : public std::logic_error
{
public:
    NotImplementedException()
        : std::logic_error("Not implemented exception"), _text("")
    {}
    NotImplementedException(std::string customText)
        : std::logic_error("Not implemented exception"), _text(customText)
    {}

    virtual const char* what() const noexcept
	{
    	return this->_text.data();
	}

private:
    std::string _text;
};

} /* namespace ddj */
#endif /* DDJ_NOT_IMPLEMENTED_EXCEPTION_HPP_ */
