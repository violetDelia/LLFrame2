#include "test_config.hpp"
#ifdef TEST_BLAS_SUM
#include <gtest/gtest.h>
#include "test_common.hpp"
#include "blas/blas.hpp"
template <class Ty, class Device>
void test_blas_adapter_sum_for_each_device() {
    using Blas_adapter = llframe::blas::Blas_Adapter<Device>;
    IS_SAME(Device, llframe::device::GPU) {
        return;
    }
    for (int i = 1; i < 32; i++) {
        size_t n = (size_t{1} << i) / sizeof(Ty);
        if (n < 2) continue;
        Ty *x = new Ty[n];
        for (int i = 0; i < n; i++) { x[i] = 0; }
        x[0] = Ty{1};
        x[n - 1] = Ty{1};
        if constexpr (std::numeric_limits<Ty>::is_signed) { x[n - 1] = Ty{-1}; }
        if constexpr (!llframe::blas::is_Support_Openblas<Device, Ty>) {
            ASSERT_THROW(Blas_adapter::sum(n, x, 1),
                         llframe::exception::Unimplement);
            delete[] x;
            return;
        }
        Ty *null = nullptr;
        ASSERT_THROW(Blas_adapter::sum(n, null, 1),
                     llframe::exception::Null_Pointer);
        ASSERT_THROW(Blas_adapter::sum(n, x, -1),
                     llframe::exception::Bad_Parameter);
        ASSERT_EQ(Blas_adapter::sum(n, x, 1), 0);
        ASSERT_EQ(Blas_adapter::sum(n / 2, x, 2), 1);
        delete[] x;
    }
};

template <class Ty>
void test_blas_adapter_sum() {
    APPLY_TUPLE_2(Device_Tuple, Ty, test_blas_adapter_sum_for_each_device);
}

TEST(Blas_adapter, sum) {
    APPLY_TUPLE(Arithmetic_Tuple, test_blas_adapter_sum);
}
#endif // TEST_BLAS_SUM