#include "test_config.hpp"
#ifdef TEST_BLAS_GEMM
#include <gtest/gtest.h>
#include "test_common.hpp"
#include "blas/blas.hpp"
template <class Ty, class Device>
void test_blas_adapter_gemm_for_each_device() {
    using Blas_adapter = llframe::blas::Blas_Adapter<Device>;
    for (int i = 1; i < 13; i++) {
        size_t n = (size_t{1} << i) / sizeof(Ty);
        if (n <= 2) continue;
        Ty *a = new Ty[n];
        Ty *x = new Ty[n * n];
        Ty *y = new Ty[n];
        for (int i = 0; i < n; i++) {
            a[i] = 1;
            x[i] = 1;
            y[i] = 1;
        }
        for (int i = 0; i < n * n; i++) { x[i] = 1; }
        if constexpr (!llframe::blas::is_Support_Openblas<Device, Ty>
                      && !llframe::blas::is_Support_Cublas<Device, Ty>) {
            ASSERT_THROW(
                Blas_adapter ::gemm(llframe::blas::Blas_Layout::Row_Major,
                                    llframe::blas::Blas_Transpose::NoTrans,
                                    llframe::blas::Blas_Transpose::NoTrans, 1,
                                    n, n, 2, a, n, x, n, 1, y, n),
                llframe::exception::Unimplement);
            delete[] a;
            delete[] x;
            delete[] y;
            return;
        }
        Ty *null = nullptr;
        ASSERT_THROW(Blas_adapter::gemm(llframe::blas::Blas_Layout::Row_Major,
                                        llframe::blas::Blas_Transpose::NoTrans,
                                        llframe::blas::Blas_Transpose::NoTrans,
                                        1, n, n, 2, a, n, x, n, n, null, n),
                     llframe::exception::Null_Pointer);
        ASSERT_THROW(Blas_adapter::gemm(llframe::blas::Blas_Layout::Row_Major,
                                        llframe::blas::Blas_Transpose::NoTrans,
                                        llframe::blas::Blas_Transpose::NoTrans,
                                        1, n, n, 2, a, n - 1, x, n, n, y, n),
                     llframe::exception::Bad_Parameter);
        ASSERT_THROW(Blas_adapter::gemm(llframe::blas::Blas_Layout::Row_Major,
                                        llframe::blas::Blas_Transpose::NoTrans,
                                        llframe::blas::Blas_Transpose::NoTrans,
                                        1, n, n, 2, a, n, x, n, 1, y, n - 1),
                     llframe::exception::Bad_Parameter);
        IS_SAME(Device, llframe::device::CPU) {
            Blas_adapter::gemm(llframe::blas::Blas_Layout::Row_Major,
                               llframe::blas::Blas_Transpose::NoTrans,
                               llframe::blas::Blas_Transpose::NoTrans, 1, n, n,
                               2, a, n, x, n, 1, y, n);
            for (int i = 0; i < n; i++) { ASSERT_EQ(y[i], 2 * n + 1); }
        }
        else {
            Ty *gpu_a;
            Ty *gpu_x;
            Ty *gpu_y;
            cudaMalloc(&gpu_x, sizeof(Ty) * n * n);
            cudaMemcpy(gpu_x, x, sizeof(Ty) * n * n, cudaMemcpyHostToDevice);
            cudaMalloc(&gpu_y, sizeof(Ty) * n);
            cudaMemcpy(gpu_y, y, sizeof(Ty) * n, cudaMemcpyHostToDevice);
            cudaMalloc(&gpu_a, sizeof(Ty) * n);
            cudaMemcpy(gpu_a, a, sizeof(Ty) * n, cudaMemcpyHostToDevice);
            Blas_adapter::gemm(llframe::blas::Blas_Layout::Row_Major,
                               llframe::blas::Blas_Transpose::NoTrans,
                               llframe::blas::Blas_Transpose::NoTrans, 1, n, n,
                               2, gpu_a, n, gpu_x, n, 1, gpu_y, n);
            cudaMemcpy(y, gpu_y, sizeof(Ty) * n, cudaMemcpyDeviceToHost);
            for (int i = 0; i < n; i++) { ASSERT_EQ(y[i], 2 * n + 1); }
            cudaFree(gpu_x);
            cudaFree(gpu_y);
            cudaFree(gpu_a);
        }
        delete[] x;
        delete[] y;
        delete[] a;
    }
};

template <class Ty>
void test_blas_adapter_gemm() {
    APPLY_TUPLE_2(Device_Tuple, Ty, test_blas_adapter_gemm_for_each_device);
}

TEST(Blas_adapter, gemm) {
    APPLY_TUPLE(Arithmetic_Tuple, test_blas_adapter_gemm);
}
#endif // TEST_BLAS_GEMM