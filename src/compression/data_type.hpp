/*
 *  data_type.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_DATA_TYPE_HPP_
#define DDJ_DATA_TYPE_HPP_

#include "core/not_implemented_exception.hpp"

namespace ddj {

enum class DataType {
    d_int,
    d_float
};

inline size_t GetDataTypeSize(DataType type)
{
	switch(type){
		case DataType::d_int: return sizeof(int);
		case DataType::d_float: return sizeof(float);
		default: throw NotImplementedException("This type is not implemented");
	}
}

template<typename T> inline DataType GetDataType();
template<> inline DataType GetDataType<int>() { return DataType::d_int; }
template<> inline DataType GetDataType<float>() { return DataType::d_float; }

} /* namespace ddj */
#endif /* DDJ_DATA_TYPE_HPP_ */
