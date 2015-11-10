/*
 * helper_print.hpp 07-11-2015 Karol Dzitkowski
 */
#ifndef DDJ_HELPER_FLOAT_HPP_
#define DDJ_HELPER_FLOAT_HPP_

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

#endif
