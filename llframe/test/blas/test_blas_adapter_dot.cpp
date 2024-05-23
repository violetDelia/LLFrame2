#include "test_config.hpp"
#ifdef TEST_BLAS_DOT
#include <gtest/gtest.h>
#include "test_common.hpp"
#include "blas/blas.hpp"
template <class Ty, class Device>
void test_blas_adapter_dot_for_each_device() {
    using Blas_adapter = llframe::blas::Blas_Adapter<Device>;
    ASSERT_DEVICE_IS_VALID(Device, 0);
    auto &device = llframe::device::Device_Platform<Device>::get_device(0);
    for (int i = 1; i < 24; i++) {
        size_t n = (size_t{1} << i) / sizeof(Ty);
        if (n <= 2) continue;
        Ty *x = new Ty[n];
        Ty *y = new Ty[n];
        for (int i = 0; i < n; i++) {
            x[i] = 0;
            y[i] = 0;
        }
        x[0] = Ty{1};
        x[n - 1] = Ty{2};
        y[0] = Ty{1};
        y[n - 1] = Ty{2};
        if constexpr (std::numeric_limits<Ty>::is_signed) {
            x[n - 1] = Ty{-2};
            y[n - 1] = Ty{-2};
        }
        if constexpr (!llframe::blas::is_Support_Openblas<Device, Ty>
                      && !llframe::blas::is_Support_Cublas<Device, Ty>) {
            ASSERT_THROW(Blas_adapter::dot(n, x, 1, y, 1, device),
                         llframe::exception::Unimplement);
            delete[] x;
            delete[] y;
            return;
        }
        Ty *null = nullptr;
        ASSERT_THROW(Blas_adapter::dot(n, null, 1, y, 1, device),
                     llframe::exception::Null_Pointer);
        ASSERT_THROW(Blas_adapter::dot(n, x, 1, y, -1, device),
                     llframe::exception::Bad_Parameter);
        IS_SAME(Device, llframe::device::CPU) {
            ASSERT_EQ(Blas_adapter::dot(n, x, 1, y, 1, device), 5);
            ASSERT_EQ(Blas_adapter::dot(n / 2, x, 2, y, 2, device), 1);
        }
        IS_SAME(Device, llframe::device::GPU) {
            Ty *gpu_x;
            Ty *gpu_y;
            if (cudaMalloc(&gpu_x, sizeof(Ty) * n)) break;
            if (cudaMemcpy(gpu_x, x, sizeof(Ty) * n, cudaMemcpyHostToDevice))
                break;
            if (cudaMalloc(&gpu_y, sizeof(Ty) * n)) break;
            if (cudaMemcpy(gpu_y, y, sizeof(Ty) * n, cudaMemcpyHostToDevice))
                break;
            if (gpu_x == nullptr || gpu_y == nullptr) {
                std::cout << "GPU memory size full!" << std::endl;
                break;
            }
            ASSERT_EQ(Blas_adapter::dot(n, gpu_x, 1, gpu_y, 1, device), 5);
            ASSERT_EQ(Blas_adapter::dot(n / 2, gpu_x, 2, gpu_y, 2, device), 1);
            cudaFree(gpu_x);
            cudaFree(gpu_y);
        }
        delete[] x;
        delete[] y;
    }
};

template <class Ty>
void test_blas_adapter_dot() {
    APPLY_TUPLE_2(Device_Tuple, Ty, test_blas_adapter_dot_for_each_device);
}

TEST(Blas_adapter, dot) {
    APPLY_TUPLE(Arithmetic_Tuple, test_blas_adapter_dot);
}
#endif // TEST_BLAS_DOT