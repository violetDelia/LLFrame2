#include "test_config.hpp"
#ifdef TEST_BLAS_GEMV
#include <gtest/gtest.h>
#include "test_common.hpp"
#include "blas/blas.hpp"
template <class Ty, class Device>
void test_blas_adapter_gemv_for_each_device() {
    using Blas_adapter = llframe::blas::Blas_Adapter<Device>;
    ASSERT_DEVICE_IS_VALID(Device, 0);
    auto &device = llframe::device::Device_Platform<Device>::get_device(0);
    for (int i = 1; i < 24; i++) {
        size_t n = (size_t{1} << i) / sizeof(Ty);
        if (n <= 2) continue;
        Ty *a = new Ty[n];
        Ty *x = new Ty[n];
        Ty *y = new Ty[n];
        for (int i = 0; i < n; i++) {
            a[i] = 1;
            x[i] = 1;
            y[i] = 1;
        }
        if constexpr (!llframe::blas::is_Support_Openblas<Device, Ty>
                      && !llframe::blas::is_Support_Cublas<Device, Ty>) {
            ASSERT_THROW(
                Blas_adapter ::gemv(llframe::blas::Blas_Layout::Row_Major,
                                    llframe::blas::Blas_Transpose::NoTrans, 1,
                                    n, 2, a, n, x, 1, 1, y, 1, device),
                llframe::exception::Unimplement);
            delete[] a;
            delete[] x;
            delete[] y;
            return;
        }
        Ty *null = nullptr;
        ASSERT_THROW(Blas_adapter::gemv(llframe::blas::Blas_Layout::Row_Major,
                                        llframe::blas::Blas_Transpose::NoTrans,
                                        1, n, 2, null, n, x, 1, 1, y, 1,
                                        device),
                     llframe::exception::Null_Pointer);
        ASSERT_THROW(Blas_adapter::gemv(llframe::blas::Blas_Layout::Row_Major,
                                        llframe::blas::Blas_Transpose::NoTrans,
                                        1, n, 2, a, n, x, 1, 1, y, -1, device),
                     llframe::exception::Bad_Parameter);
        ASSERT_THROW(Blas_adapter::gemv(llframe::blas::Blas_Layout::Row_Major,
                                        llframe::blas::Blas_Transpose::NoTrans,
                                        1, n, 2, a, n - 1, x, 1, 1, y, -1,
                                        device),
                     llframe::exception::Bad_Parameter);
        IS_SAME(Device, llframe::device::CPU) {
            Blas_adapter::gemv(llframe::blas::Blas_Layout::Row_Major,
                               llframe::blas::Blas_Transpose::NoTrans, 1, n, 2,
                               a, n, x, 1, 1, y, 1, device);
            ASSERT_EQ(y[0], 2 * n + 1);
            Blas_adapter::gemv(llframe::blas::Blas_Layout::Row_Major,
                               llframe::blas::Blas_Transpose::Trans, 1, n, 2, a,
                               n, x, 1, 0, y, 1, device);
            for (int i = 0; i < n; i++) { ASSERT_EQ(y[i], 2); }
        }
        IS_SAME(Device, llframe::device::GPU) {
            Ty *gpu_a;
            Ty *gpu_x;
            Ty *gpu_y;
            if (cudaMalloc(&gpu_x, sizeof(Ty) * n)) break;
            if (cudaMemcpy(gpu_x, x, sizeof(Ty) * n, cudaMemcpyHostToDevice))
                break;
            if (cudaMalloc(&gpu_y, sizeof(Ty) * n)) break;
            if (cudaMemcpy(gpu_y, y, sizeof(Ty) * n, cudaMemcpyHostToDevice))
                break;
            if (cudaMalloc(&gpu_a, sizeof(Ty) * n)) break;
            if (cudaMemcpy(gpu_a, a, sizeof(Ty) * n, cudaMemcpyHostToDevice))
                break;
            if (gpu_x == nullptr || gpu_y == nullptr || gpu_a == nullptr) {
                std::cout << "GPU memory size full!" << std::endl;
                break;
            }
            Blas_adapter::gemv(llframe::blas::Blas_Layout::Row_Major,
                               llframe::blas::Blas_Transpose::NoTrans, 1, n, 2,
                               gpu_a, n, gpu_x, 1, 1, gpu_y, 1, device);
            cudaMemcpy(y, gpu_y, sizeof(Ty), cudaMemcpyDeviceToHost);
            ASSERT_EQ(y[0], 2 * n + 1);
            Blas_adapter::gemv(llframe::blas::Blas_Layout::Row_Major,
                               llframe::blas::Blas_Transpose::Trans, 1, n, 2,
                               gpu_a, n, gpu_x, 1, 0, gpu_y, 1, device);
            cudaMemcpy(y, gpu_y, sizeof(Ty) * n, cudaMemcpyDeviceToHost);
            for (int i = 0; i < n; i++) { ASSERT_EQ(y[i], 2); }
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
void test_blas_adapter_gemv() {
    APPLY_TUPLE_2(Device_Tuple, Ty, test_blas_adapter_gemv_for_each_device);
}

TEST(Blas_adapter, gemv) {
    APPLY_TUPLE(Arithmetic_Tuple, test_blas_adapter_gemv);
}
#endif // TEST_BLAS_GEMV