/*
 *  stencil_operators.hpp
 *
 *  Created on: 04-06-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_STENCIL_OPERATORS_HPP_
#define DDJ_STENCIL_OPERATORS_HPP_

namespace ddj {

template<typename T>
struct OutsideOperator
{
	T low;
	T high;

	__host__ __device__
	bool operator()(const T &input) const
	{
		if(input > high || input < low) return false;
		else return true;
	}
};

template<typename T>
struct InsideOperator
{
	T low;
	T high;

	__host__ __device__
	bool operator()(const T &input) const
	{
		if(input > high || input < low) return true;
		else return false;
	}
};

template<typename T>
struct EqualOperator
{
	T value;

	__host__ __device__
	bool operator()(const T &input) const
	{
		return input == value;
	}
};

template<typename T>
struct NotEqualOperator
{
	T value;

	__host__ __device__
	bool operator()(const T &input) const
	{
		return input != value;
	}
};

template<typename T>
struct LowerOperator
{
	T value;

	__host__ __device__
	bool operator()(const T &input) const
	{
		return input < value;
	}
};

template<typename T>
struct GreaterOperator
{
	T value;

	__host__ __device__
	bool operator()(const T &input) const
	{
		return input > value;
	}
};

} /* namespace ddj */
#endif /* DDJ_STENCIL_OPERATORS_HPP_ */
