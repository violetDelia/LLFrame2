#include "test_config.hpp"
#ifdef TEST_BLAS_SCAL
#include <gtest/gtest.h>
#include "test_common.hpp"
#include "blas/blas.hpp"
template <class Ty, class Device>
void test_blas_adapter_scal_for_each_device() {
    using Blas_adapter = llframe::blas::Blas_Adapter<Device>;
    ASSERT_DEVICE_IS_VALID(Device, 0);
    auto &device = llframe::device::Device_Platform<Device>::get_device(0);
    for (int i = 1; i < 24; i++) {
        size_t n = (size_t{1} << i) / sizeof(Ty);
        if (n <= 2) continue;
        Ty *x = new Ty[n];
        for (int i = 0; i < n; i++) { x[i] = i % 8; }
        if constexpr (!llframe::blas::is_Support_Openblas<Device, Ty>
                      && !llframe::blas::is_Support_Cublas<Device, Ty>) {
            ASSERT_THROW(Blas_adapter::scal(n, 1, x, 1, device),
                         llframe::exception::Unimplement);
            delete[] x;
            return;
        }
        Ty *null = nullptr;
        ASSERT_THROW(Blas_adapter::scal(n, 2, null, 1, device),
                     llframe::exception::Null_Pointer);
        ASSERT_THROW(Blas_adapter::scal(n, 2, x, -1, device),
                     llframe::exception::Bad_Parameter);
        IS_SAME(Device, llframe::device::CPU) {
            Blas_adapter::scal(n, 2, x, 1, device);
            for (int i = 0; i < n; i++) { ASSERT_EQ(x[i], 2 * (i % 8)); }
        }
        IS_SAME(Device, llframe::device::GPU) {
            Ty *gpu_x;
            if (cudaMalloc(&gpu_x, sizeof(Ty) * n)) break;
            if (cudaMemcpy(gpu_x, x, sizeof(Ty) * n, cudaMemcpyHostToDevice))
                break;
            Blas_adapter::scal(n, 2, gpu_x, 1, device);
            if (cudaMemcpy(x, gpu_x, sizeof(Ty) * n, cudaMemcpyDeviceToHost))
                break;
            if (gpu_x == nullptr) {
                std::cout << "GPU memory size full!" << std::endl;
                break;
            }
            for (int i = 0; i < n; i++) { ASSERT_EQ(x[i], 2 * (i % 8)); }
            cudaFree(gpu_x);
        }
        delete[] x;
    }
};

template <class Ty>
void test_blas_adapter_scal() {
    APPLY_TUPLE_2(Device_Tuple, Ty, test_blas_adapter_scal_for_each_device);
}

TEST(Blas_adapter, scal) {
    APPLY_TUPLE(Arithmetic_Tuple, test_blas_adapter_scal);
}
#endif // TEST_BLAS_SCAL