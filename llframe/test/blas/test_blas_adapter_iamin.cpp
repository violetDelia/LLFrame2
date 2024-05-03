#include "test_config.hpp"
#ifdef TEST_BLAS_IAMIN
#include <gtest/gtest.h>
#include "test_common.hpp"
#include "blas/blas.hpp"
template <class Ty, class Device>
void test_blas_adapter_iamin_for_each_device() {
    using Blas_adapter = llframe::blas::Blas_Adapter<Device>;
    for (int i = 1; i < 32; i++) {
        size_t n = (size_t{1} << i) / sizeof(Ty);
        if (n <= 2) continue;
        Ty *x = new Ty[n];
        for (int i = 0; i < n; i++) { x[i] = 127; }
        x[0] = Ty{102};
        x[n - 1] = Ty{101};
        if constexpr (std::numeric_limits<Ty>::is_signed) {
            x[0] = Ty{-102};
            x[n - 1] = Ty{-101};
        }
        if constexpr (!llframe::blas::is_Support_Openblas<Device, Ty>
                      && !llframe::blas::is_Support_Cublas<Device, Ty>) {
            ASSERT_THROW(Blas_adapter::iamin(n, x, 1),
                         llframe::exception::Unimplement);
            delete[] x;
            return;
        }
        Ty *null = nullptr;
        ASSERT_THROW(Blas_adapter::iamin(n, null, 1),
                     llframe::exception::Null_Pointer);
        ASSERT_THROW(Blas_adapter::iamin(n, x, -1),
                     llframe::exception::Bad_Parameter);
        IS_SAME(Device, llframe::device::CPU) {
            ASSERT_EQ(Blas_adapter::iamin(n, x, 1), n - 1);
            ASSERT_EQ(Blas_adapter::iamin(n / 2, x, 2), 0);
        }
        else {
            Ty *gpu_x;
            cudaMalloc(&gpu_x, sizeof(Ty) * n);
            cudaMemcpy(gpu_x, x, sizeof(Ty) * n, cudaMemcpyHostToDevice);
            ASSERT_EQ(Blas_adapter::iamin(n, gpu_x, 1), n - 1);
            ASSERT_EQ(Blas_adapter::iamin(n / 2, gpu_x, 2), 0);
            cudaFree(gpu_x);
        }
        delete[] x;
    }
};

template <class Ty>
void test_blas_adapter_iamin() {
    APPLY_TUPLE_2(Device_Tuple, Ty, test_blas_adapter_iamin_for_each_device);
}

TEST(Blas_adapter, iamin) {
    APPLY_TUPLE(Arithmetic_Tuple, test_blas_adapter_iamin);
}
#endif // TEST_BLAS_IAMIN