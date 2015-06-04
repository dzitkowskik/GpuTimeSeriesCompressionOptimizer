/*
 * operators.cuh
 *
 *  Created on: 04-06-2015
 *      Author: Karol Dzitkowski
 */

#ifndef OPERATORS_CUH_
#define OPERATORS_CUH_

template<typename T>
struct outsideOperator
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

#endif /* OPERATORS_CUH_ */
