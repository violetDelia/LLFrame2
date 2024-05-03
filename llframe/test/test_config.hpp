#ifndef __LLFRAME_TEST_CONFIG__
#define __LLFRAME_TEST_CONFIG__
// #define TEST_CONFIG
// #define TEST_BASE_TYPE
// #define TEST_EXCEPTION
// #define TEST_SHAPE
// #define TEST_DEVICE
// #define TEST_ALLOCATOR

#define TEST_BLAS

#ifdef TEST_BLAS
#define TEST_BLAS_LEVER1
#endif // TEST_BLAS

#ifdef TEST_BLAS_LEVER1
#define TEST_BLAS_SUM
#define TEST_BLAS_ASUM
#define TEST_BLAS_DOT
#endif // TEST_BLAS_LEVER1
#endif //__LLFRAME_TEST_CONFIG__