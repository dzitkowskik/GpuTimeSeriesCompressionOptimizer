/*
 * operators.cuh
 *
 *  Created on: 04-06-2015
 *      Author: Karol Dzitkowski
 */

#ifndef OPERATORS_CUH_
#define OPERATORS_CUH_

namespace ddj {

template<typename InputType, typename OutputType>
struct OutsideOperator
{
	InputType low;
	InputType high;

	__host__ __device__
	OutputType operator()(const InputType &value) const
	{
		if(value > high || value < low) return false;
		else return true;
	}
};

template<typename InputType, typename OutputType>
struct InsideOperator
{
	InputType low;
	InputType high;

	__host__ __device__
	OutputType operator()(const InputType &value) const
	{
		if(value > high || value < low) return true;
		else return false;
	}
};

template<typename InputType, typename OutputType>
struct ModulusOperator
{
	InputType mod;

    __host__ __device__
    OutputType operator()(const InputType &x) { return x % mod; }
};

template<typename InputType, typename OutputType>
struct AdditionOperator
{
	InputType value;

	__host__ __device__
	OutputType operator()(const InputType &x) { return x + value; }
};

template<typename InputType, typename OutputType>
struct SubtractionOperator
{
	InputType value;

	__host__ __device__
	OutputType operator()(const InputType &x) { return x - value; }
};

template<typename InputType, typename OutputType>
struct MultiplicationOperator
{
	InputType value;

	__host__ __device__
	OutputType operator()(const InputType &x) { return x * value; }
};

template<typename InputType, typename OutputType>
struct DivisionOperator
{
	InputType value;

	__host__ __device__
	OutputType operator()(const InputType &x) { return (OutputType)x / value; }
};

template<typename InputType, typename OutputType>
struct AbsoluteOperator
{
	__host__ __device__
	OutputType operator()(const InputType &x) { return x > 0 ? x : -x; }
};

template<typename InputType, typename OutputType>
struct ZeroOperator
{
	__host__ __device__
	OutputType operator()(const InputType &x) { return 0; }
};

template<typename InputType, typename OutputType>
struct OneOperator
{
	__host__ __device__
	OutputType operator()(const InputType &x) { return 1; }
};

template<typename InputType, typename OutputType>
struct NegateOperator
{
	__host__ __device__
	OutputType operator()(const InputType &x) { return !x; }
};

template<typename InputType, typename OutputType>
struct FillOperator
{
	InputType value;

	__host__ __device__
	OutputType operator()(const InputType &x) { return value; }
};

template<typename InputType, typename OutputType>
struct SetPrecisionOperator
{
	int precision;

	__host__ __device__
	OutputType operator()(const InputType &x)
	{
		int mul = pow(10, precision);
		long long int tmp = round(x * mul);
		return (OutputType)tmp / mul;
	}
};

template<typename InputType, typename OutputType>
struct FloatingPointToIntegerOperator
{
	int precision;

	__host__ __device__
	OutputType operator()(const InputType &x)
	{
		return lrint(x * pow(10, precision));
	}
};

template<typename InputType, typename OutputType>
struct IntegerToFloatingPointOperator
{
	int precision;

	__host__ __device__
	OutputType operator()(const InputType &x)
	{
		return (OutputType)x / pow(10, precision);
	}
};

} /* namespace ddj */
#endif /* OPERATORS_CUH_ */
