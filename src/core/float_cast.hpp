/*
 * float_cast.hpp 07-11-2015 Karol Dzitkowski
 */
#ifndef DDJ_CORE_FLOAT_CAST_HPP_
#define DDJ_CORE_FLOAT_CAST_HPP_

namespace ddj
{

union floatCastUnion {
  float value;
  struct {
    unsigned int mantisa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
};

union doubleCastUnion {
  double value;
  struct {
    unsigned long int mantisa : 52;
    unsigned int exponent : 11;
    unsigned int sign : 1;
  } parts;
};

} /* namespace ddj */
#endif /* DDJ_CORE_FLOAT_CAST_HPP_ */
