// #ifdef DDJ_HELPER_MACROS_H_
// #define DDJ_HELPER_MACROS_H_

// namespace ddj
// {

#define CUDA_CHECK_RETURN(value) {                               \
    cudaError_t _m_cudaStat = value;                             \
    if (_m_cudaStat != cudaSuccess) {                            \
        fprintf(stderr, "Error %s at line %d in file %s\n",      \
            cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
        exit(1);                                                 \
    } }

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);}} while(0)


// } /* namespace ddj */
// #endif /* DDJ_HELPER_MACROS_H_ */
