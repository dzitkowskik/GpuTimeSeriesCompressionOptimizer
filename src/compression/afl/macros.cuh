#ifndef macros
#define macros 0

// This should work independently from _CUDA_ARCH__ number

#define NBITSTOMASK(n) ((1<<(n)) - 1)
#define LNBITSTOMASK(n) ((1L<<(n)) - 1)

#define fillto(b,c) (((c + b - 1) / b) * b)

#define _unused(x) x __attribute__((unused))
#define convert_struct(n, s)  struct sgn {signed int x:n;} __attribute__((unused)) s

//TODO: distinguish between signed/unsigned versions

// This depend on _CUDA_ARCH__ number
__device__ __host__ __forceinline__ unsigned int GETNPBITS( int source, unsigned int num_bits, unsigned int bit_start)
{
#if __CUDA_ARCH__ > 200  // This improves performance
    unsigned int bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"((unsigned int) source), "r"(bit_start), "r"(num_bits));
    return bits;
#else
    return ((source>>bit_start) & NBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned int GETNPBITS( unsigned int source, unsigned int num_bits, unsigned int bit_start)
{
#if __CUDA_ARCH__ > 200  // This improves performance
    unsigned int bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"((unsigned int) source), "r"(bit_start), "r"(num_bits));
    return bits;
#else
    return ((source>>bit_start) & NBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned long GETNPBITS( long source, unsigned int num_bits, unsigned int bit_start)
{
#if __CUDA_ARCH__ > 200  // This improves performance
    unsigned long bits;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(bits) : "l"((unsigned long) source), "r"(bit_start), "r"(num_bits));
    return bits;
#else
    return ((source>>bit_start) & LNBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned long GETNPBITS( unsigned long source, unsigned int num_bits, unsigned int bit_start)
{
#if __CUDA_ARCH__ > 200  // This improves performance
    unsigned long bits;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(bits) : "l"((unsigned long) source), "r"(bit_start), "r"(num_bits));
    return bits;
#else
    return ((source>>bit_start) & LNBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned long GETNBITS( long source, unsigned int num_bits)
{
#if __CUDA_ARCH__ > 200  // Use bfe implementation
    return GETNPBITS(source, num_bits, 0);
#else // In other case this will be faster
    return ((source) & LNBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned long GETNBITS( unsigned long source, unsigned int num_bits)
{
#if __CUDA_ARCH__ > 200  // Use bfe implementation
    return GETNPBITS(source, num_bits, 0);
#else // In other case this will be faster
    return ((source) & LNBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned int GETNBITS( int source, unsigned int num_bits)
{
#if __CUDA_ARCH__ > 200  // Use bfe implementation
    return GETNPBITS(source, num_bits, 0);
#else // In other case this will be faster
    return ((source) & NBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned int GETNBITS( unsigned int source, unsigned int num_bits)
{
#if __CUDA_ARCH__ > 200  // Use bfe implementation
    return GETNPBITS(source, num_bits, 0);
#else // In other case this will be faster
    return ((source) & NBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned int BITLEN(unsigned int word)
{
    unsigned int ret=0;
#if __CUDA_ARCH__ > 200
    asm volatile ("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
#else
    while (word >>= 1)
      ret++;
#endif
   return ret;
}

__host__ __device__
inline int ALT_BITLEN(int v)
{
    register unsigned int r; // result of log2(v) will go here
    register unsigned int shift;

    r =     (v > 0xFFFF) << 4; v >>= r;
    shift = (v > 0xFF  ) << 3; v >>= shift; r |= shift;
    shift = (v > 0xF   ) << 2; v >>= shift; r |= shift;
    shift = (v > 0x3   ) << 1; v >>= shift; r |= shift;
    r |= (v >> 1);
    return r+1;
}

#define SGN(a) (int)((unsigned int)((int)a) >> (sizeof(int) * CHAR_BIT - 1))
#define GETNSGNBITS(a,n,b) ((SGN(a) << (n-1)) | GETNBITS(((a)>>(b-n)), (n-1)))

// Make a FOREACH macro
#define FE_1(WHAT, X) WHAT(X)
#define FE_2(WHAT, X, ...) WHAT(X)FE_1(WHAT, __VA_ARGS__)
#define FE_3(WHAT, X, ...) WHAT(X)FE_2(WHAT, __VA_ARGS__)
#define FE_4(WHAT, X, ...) WHAT(X)FE_3(WHAT, __VA_ARGS__)
#define FE_5(WHAT, X, ...) WHAT(X)FE_4(WHAT, __VA_ARGS__)
#define FE_6(WHAT, X, ...) WHAT(X)FE_5(WHAT, __VA_ARGS__)
//... repeat as needed

#define GET_MACRO(_1,_2,_3,_4,_5,NAME,...) NAME
#define FOR_EACH(action,...) \
  GET_MACRO(__VA_ARGS__,FE_5,FE_4,FE_3,FE_2,FE_1)(action,__VA_ARGS__)

#endif
