/*
 * operators.cuh
 *
 *  Created on: 04-06-2015
 *      Author: Karol Dzitkowski
 */

#ifndef OPERATORS_CUH_
#define OPERATORS_CUH_

template<typename T>
struct OutsideOperator
{
	T low;
	T high;

	__host__ __device__
	bool operator()(const T &value) const
	{
		if(value > high || value < low) return false;
		else return true;
	}
};

template<typename T>
struct InsideOperator
{
	T low;
	T high;

	__host__ __device__
	bool operator()(const T &value) const
	{
		if(value > high || value < low) return true;
		else return false;
	}
};

template<typename T>
struct ModulusOperator
{
    T mod;

    __host__ __device__
    T operator()(const T &x) { return x % mod; }
};

template<typename T>
struct AdditionOperator
{
	T value;

	__host__ __device__
	T operator()(const T &x) { return x + value; }
};

template<typename T>
struct SubtractionOperator
{
	T value;

	__host__ __device__
	T operator()(const T &x) { return x - value; }
};

template<typename T>
struct MultiplicationOperator
{
	T value;

	__host__ __device__
	T operator()(const T &x) { return x * value; }
};

template<typename T>
struct DivisionOperator
{
	T value;

	__host__ __device__
	T operator()(const T &x) { return x / value; }
};

template<typename T>
struct AbsoluteOperator
{
	__host__ __device__
	T operator()(const T &x) { return x > 0 ? x : -x; }
};

#endif /* OPERATORS_CUH_ */
