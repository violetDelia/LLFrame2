#include "test_config.hpp"
#ifdef TEST_BLAS_COPY
#include <gtest/gtest.h>
#include "test_common.hpp"
#include "blas/blas.hpp"
template <class Ty, class Device>
void test_blas_adapter_copy_for_each_device() {
    using Blas_adapter = llframe::blas::Blas_Adapter<Device>;
    for (int i = 1; i < 32; i++) {
        size_t n = (size_t{1} << i) / sizeof(Ty);
        if (n <= 2) continue;
        Ty *x = new Ty[n];
        Ty *y = new Ty[n];
        for (int i = 0; i < n; i++) { x[i] = i % 128; }
        if constexpr (!llframe::blas::is_Support_Openblas<Device, Ty>
                      && !llframe::blas::is_Support_Cublas<Device, Ty>) {
            ASSERT_THROW(Blas_adapter::copy(n, x, 1, y, 1),
                         llframe::exception::Unimplement);
            delete[] x;
            delete[] y;
            return;
        }
        Ty *null = nullptr;
        ASSERT_THROW(Blas_adapter::copy(n, null, 1, y, 1),
                     llframe::exception::Null_Pointer);
        ASSERT_THROW(Blas_adapter::copy(n, x, 1, y, -1),
                     llframe::exception::Bad_Parameter);
        IS_SAME(Device, llframe::device::CPU) {
            Blas_adapter::copy(n, x, 1, y, 1);
            for (int i = 0; i < n; i++) { ASSERT_EQ(y[i], x[i]); }
        }
        else {
            Ty *gpu_x;
            Ty *gpu_y;
            cudaMalloc(&gpu_x, sizeof(Ty) * n);
            cudaMemcpy(gpu_x, x, sizeof(Ty) * n, cudaMemcpyHostToDevice);
            cudaMalloc(&gpu_y, sizeof(Ty) * n);
            cudaMemcpy(gpu_y, y, sizeof(Ty) * n, cudaMemcpyHostToDevice);
            Blas_adapter::copy(n, gpu_x, 1, gpu_y, 1);
            cudaMemcpy(y, gpu_y, sizeof(Ty) * n, cudaMemcpyDeviceToHost);
            for (int i = 0; i < n; i++) { ASSERT_EQ(y[i], x[i]); }
            cudaFree(gpu_x);
            cudaFree(gpu_y);
        }
        delete[] x;
        delete[] y;
    }
};

template <class Ty>
void test_blas_adapter_copy() {
    APPLY_TUPLE_2(Device_Tuple, Ty, test_blas_adapter_copy_for_each_device);
}

TEST(Blas_adapter, copy) {
    APPLY_TUPLE(Arithmetic_Tuple, test_blas_adapter_copy);
}
#endif // TEST_BLAS_COPY