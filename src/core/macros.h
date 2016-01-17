#include <stdio.h>

#define CPY_DTD cudaMemcpyDeviceToDevice
#define CPY_DTH cudaMemcpyDeviceToHost
#define CPY_HTD cudaMemcpyHostToDevice
#define CPY_HTH cudaMemcpyHostToHost

#define CUDA_ASSERT_RETURN(value) {                              \
    cudaError_t _m_cudaStat = value;                             \
    if (_m_cudaStat != cudaSuccess) {                            \
        fprintf(stderr, "Error %s at line %d in file %s\n",      \
            cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
        exit(1);                                                 \
    } }

#define CUDA_CALL(x) do { cudaError_t err = x; if((err)!=cudaSuccess) { \
    printf("Error at %s:%d - %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); }} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d:error=%d\n",__FILE__,__LINE__,x);}} while(0)

#define SUPPORTED_DATA_TYPES char, bool, short, int, unsigned, long, long long, float, double

// Make a FOREACH macro
#define FE_1(WHAT, X) WHAT(X)
#define FE_2(WHAT, X, ...) WHAT(X)FE_1(WHAT, __VA_ARGS__)
#define FE_3(WHAT, X, ...) WHAT(X)FE_2(WHAT, __VA_ARGS__)
#define FE_4(WHAT, X, ...) WHAT(X)FE_3(WHAT, __VA_ARGS__)
#define FE_5(WHAT, X, ...) WHAT(X)FE_4(WHAT, __VA_ARGS__)
#define FE_6(WHAT, X, ...) WHAT(X)FE_5(WHAT, __VA_ARGS__)
#define FE_7(WHAT, X, ...) WHAT(X)FE_6(WHAT, __VA_ARGS__)
#define FE_8(WHAT, X, ...) WHAT(X)FE_7(WHAT, __VA_ARGS__)
#define FE_9(WHAT, X, ...) WHAT(X)FE_8(WHAT, __VA_ARGS__)
//... repeat as needed

#define GET_MACRO(_1,_2,_3,_4,_5,_6,_7,_8,NAME,...) NAME
#define FOR_EACH(action,...) \
  GET_MACRO(__VA_ARGS__,FE_8,FE_7,FE_6,FE_5,FE_4,FE_3,FE_2,FE_1)(action,__VA_ARGS__)
