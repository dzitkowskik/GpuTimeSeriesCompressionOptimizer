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
    T operator()(T x) { return x % mod; }
};


#endif /* OPERATORS_CUH_ */
