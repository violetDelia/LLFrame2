#include "test_config.hpp"
#ifdef TEST_BLAS_ASUM
#include <gtest/gtest.h>
#include "test_common.hpp"
#include "blas/blas.hpp"
template <class Ty, class Device>
void test_blas_adapter_asum_for_each_device() {
    using Blas_adapter = llframe::blas::Blas_Adapter<Device>;
    for (int i = 1; i < 32; i++) {
        size_t n = (size_t{1} << i) / sizeof(Ty);
        if (n <= 2) continue;
        Ty *x = new Ty[n];
        for (int i = 0; i < n; i++) { x[i] = 0; }
        x[0] = Ty{1};
        x[n - 1] = Ty{1};
        if constexpr (!llframe::blas::is_Support_Openblas<Device, Ty>
                      && !llframe::blas::is_Support_Cublas<Device, Ty>) {
            ASSERT_THROW(Blas_adapter::asum(n, x, 1),
                         llframe::exception::Unimplement);
            delete[] x;
            return;
        } else {
            x[n - 1] = Ty{-1};
        }
        IS_SAME(Device, llframe::device::CPU) {
            ASSERT_EQ(Blas_adapter::asum(n, x, 1), 2);
            ASSERT_EQ(Blas_adapter::asum(n / 2, x, 2), 1);
        }
        else {
            Ty *gpu_x;
            cudaMalloc(&gpu_x, sizeof(Ty) * n);
            cudaMemcpy(gpu_x, x, sizeof(Ty) * n, cudaMemcpyHostToDevice);
            ASSERT_EQ(Blas_adapter::asum(n, gpu_x, 1), 2);
            ASSERT_EQ(Blas_adapter::asum(n / 2, gpu_x, 2), 1);
            cudaFree(gpu_x);
        }
        delete[] x;
    }
};

template <class Ty>
void test_blas_adapter_asum() {
    APPLY_TUPLE_2(Device_Tuple, Ty, test_blas_adapter_asum_for_each_device);
}

TEST(Blas_adapter, asum) {
    APPLY_TUPLE(Arithmetic_Tuple, test_blas_adapter_asum);
}
#endif // TEST_BLAS_ASUM