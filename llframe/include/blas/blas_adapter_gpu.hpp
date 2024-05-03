//    Copyright 2023 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

/**
 * @brief 线性代数库调用转接器
 * @remarks GPU 版本
 */
#ifndef __LLFRAME_BLAS_ADAPTER_GPU_HPP__
#define __LLFRAME_BLAS_ADAPTER_GPU_HPP__
#include "blas/blas_adapter.hpp"
#include "device/device_impl.hpp"
#include "device/device_platform.hpp"
namespace llframe::blas {
template <>
class Blas_Adapter<device::GPU> : public _Blas_Adapter_Base<device::GPU> {
public:
    using Self = Blas_Adapter<device::GPU>;
    using Base = _Blas_Adapter_Base<device::GPU>;

    using size_type = typename Base::size_type;
    using difference_type = typename Base::difference_type;
    using const_dif_t = typename Base::const_dif_t;
    using device_type = typename Base::device_type;

    using plat = typename Base::plat;

    using Layout = typename Base::Layout;
    using Transpose = typename Base::Transpose;
    using Uplo = typename Base::Uplo;
    using Diag = typename Base::Diag;
    using Side = typename Base::Side;

protected:
    static constexpr const cublasOperation_t convert_(const Transpose trans) {
        if constexpr (trans == Transpose::NoTrans)
            return cublasOperation_t::CUBLAS_OP_N;
        if constexpr (trans == Transpose::Trans)
            return cublasOperation_t::CUBLAS_OP_T;
        __LLFRAME_THROW_EXCEPTION_INFO__(exception::Bad_Parameter,
                                         "cant not convert transpose!")
    }

public:
    /**
     * @brief 向量x的绝对值和
     */
    @remarks GPU 版本 template <is_Arithmetic X>
    static constexpr X asum(const_dif_t n, const X *x, const_dif_t incx) {
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            X res{};
            cublasSasum_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), x,
                           static_cast<const int>(incx), &res);
            return res;
        }
        if constexpr (is_Same_Ty<double, X>) {
            X res{};
            cublasDasum_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), x,
                           static_cast<const int>(incx), &res);
            return res;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief res = sum(xi*yi)
     */
    @remarks GPU 版本 template <is_Arithmetic X, is_Arithmetic Y>
    static constexpr X dot(const_dif_t n, const X *x, const_dif_t incx,
                           const Y *y, const_dif_t incy) {
        ensure_no_null_pointer_(x, y);
        ensure_not_negative_<const int>(n, incx, incy);
        if constexpr (is_Same_Ty<float, X, Y>) {
            X res{};
            cublasSdot_v2(plat::get_active_device().cublas_handle(),
                          static_cast<const int>(n), x,
                          static_cast<const int>(incx), y,
                          static_cast<const int>(incy), &res);
            return res;
        }
        if constexpr (is_Same_Ty<double, X, Y>) {
            X res{};
            cublasDdot_v2(plat::get_active_device().cublas_handle(),
                          static_cast<const int>(n), x,
                          static_cast<const int>(incx), y,
                          static_cast<const int>(incy), &res);
            return res;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief res = ||x||^2
     * @remarks GPU 版本
     */
    template <is_Arithmetic X>
    static constexpr X nrm2(const_dif_t n, const X *x, const_dif_t incx) {
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            X res{};
            cublasSnrm2_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), x,
                           static_cast<const int>(incx), &res);
            return res;
        }
        if constexpr (is_Same_Ty<double, X>) {
            X res{};
            cublasDnrm2_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), x,
                           static_cast<const int>(incx), &res);
            return res;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 绝对值最大的第一个索引
     * @remarks GPU 版本
     */
    template <is_Arithmetic X>
    static constexpr difference_type iamax(const_dif_t n, const X *x,
                                           const_dif_t incx) {
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            difference_type res{};
            cublasIsamax_v2(plat::get_active_device().cublas_handle(),
                            static_cast<const int>(n), x,
                            static_cast<const int>(incx), &res);
            return res;
        }
        if constexpr (is_Same_Ty<double, X>) {
            difference_type res{};
            cublasIdamax_v2(plat::get_active_device().cublas_handle(),
                            static_cast<const int>(n), x,
                            static_cast<const int>(incx), &res);
            return res;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 绝对值最小的第一个索引
     * @remarks GPU 版本
     */
    template <is_Arithmetic X>
    static constexpr difference_type iamin(const_dif_t n, const X *x,
                                           const_dif_t incx) {
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            difference_type res{};
            cublasIsamin_v2(plat::get_active_device().cublas_handle(),
                            static_cast<const int>(n), x,
                            static_cast<const int>(incx), &res);
            return res;
        }
        if constexpr (is_Same_Ty<double, X>) {
            difference_type res{};
            cublasIdamin_v2(plat::get_active_device().cublas_handle(),
                            static_cast<const int>(n), x,
                            static_cast<const int>(incx), &res);
            return res;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief y = alpha*x+y
     * @remarks GPU 版本
     */
    template <is_Arithmetic X, is_Arithmetic Y, is_Arithmetic Alpha>
    static constexpr void axpy(const_dif_t n, const Alpha alpha, const X *x,
                               const_dif_t incx, Y *y, const_dif_t incy) {
        ensure_no_null_pointer_(x, y);
        ensure_not_negative_<const int>(n, incx, incy);
        if constexpr (is_Same_Ty<float, X, Y>) {
            cublasCaxpy_v2(
                plat::get_active_device().cublas_handle(),
                static_cast<const int>(n), static_cast<const X>(alpha), x,
                static_cast<const int>(incx), y, static_cast<const int>(incy));
            return;
        }
        if constexpr (is_Same_Ty<double, X, Y>) {
            cublasDaxpy_v2(
                plat::get_active_device().cublas_handle(),
                static_cast<const int>(n), static_cast<const X>(alpha), x,
                static_cast<const int>(incx), y, static_cast<const int>(incy));
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief y = x
     * @remarks GPU 版本
     */
    template <is_Arithmetic X, is_Arithmetic Y>
    static constexpr void copy(const_dif_t n, const X *x, const_dif_t incx,
                               Y *y, const_dif_t incy) {
        ensure_no_null_pointer_(x, y);
        ensure_not_negative_<const int>(n, incx, incy);
        if constexpr (is_Same_Ty<float, X, Y>) {
            cublasCcopy_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), x,
                           static_cast<const int>(incx), y,
                           static_cast<const int>(incy));
            return;
        }
        if constexpr (is_Same_Ty<double, X, Y>) {
            cublasDcopy_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), x,
                           static_cast<const int>(incx), y,
                           static_cast<const int>(incy));
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief y = x;x = y
     * @remarks GPU 版本
     */
    template <is_Arithmetic X, is_Arithmetic Y>
    static constexpr void swap(const_dif_t n, X *x, const_dif_t incx, Y *y,
                               const_dif_t incy) {
        ensure_no_null_pointer_(x, y);
        ensure_not_negative_<const int>(n, incx, incy);
        if constexpr (is_Same_Ty<float, X, Y>) {
            cublasCswap_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), x,
                           static_cast<const int>(incx), y,
                           static_cast<const int>(incy));
            return;
        }
        if constexpr (is_Same_Ty<double, X, Y>) {
            cublasDswap_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), x,
                           static_cast<const int>(incx), y,
                           static_cast<const int>(incy));
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief x = alpha * x
     * @remarks GPU 版本
     */
    template <is_Arithmetic X, is_Arithmetic Alpha>
    static constexpr void scal(const_dif_t n, const Alpha alpha, X *x,
                               const_dif_t incx) {
        __THROW_UNIMPLEMENTED__;
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            cublasSscal_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n),
                           static_cast<const X>(alpha), x,
                           static_cast<const int>(incx));
            return;
        }
        if constexpr (is_Same_Ty<double, X>) {
            cublasDscal_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n),
                           static_cast<const X>(alpha), x,
                           static_cast<const int>(incx));
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };

    template <is_Arithmetic X, is_Arithmetic Y, is_Arithmetic A,
              is_Arithmetic Alpha, is_Arithmetic Beta>
    static constexpr void gemv(const Layout layout, const Transpose trans,
                               const_dif_t m, const_dif_t n, const Alpha alpha,
                               const A *a, const_dif_t lda, const X *x,
                               const_dif_t incx, const Beta beta, Y *y,
                               const_dif_t incy) {
        ensure_no_null_pointer_(a, x, y);
        ensure_not_negative_<const int>(m, n, lda, incx, incy);
        if constexpr (is_Same_Ty<float, A, X, Y>) {
            cublasSgemv_v2(
                plat::get_active_device().cublas_handle(), convert_(trans),
                static_cast<const int>(m), static_cast<const int>(n),
                static_cast<const X>(alpha), a, static_cast<const int>(lda), x,
                static_cast<const int>(incx), static_cast<const X>(beta), y,
                static_cast<const int>(incy));
            return;
        }
        if constexpr (is_Same_Ty<double, A, X, Y>) {
            cublasDgemv_v2(
                plat::get_active_device().cublas_handle(), convert_(trans),
                static_cast<const int>(m), static_cast<const int>(n),
                static_cast<const X>(alpha), a, static_cast<const int>(lda), x,
                static_cast<const int>(incx), static_cast<const X>(beta), y,
                static_cast<const int>(incy));
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief a = alpha*x*y^T + a
     * @remarks GPU 版本
     */
    template <is_Arithmetic X, is_Arithmetic Y, is_Arithmetic A,
              is_Arithmetic Alpha>
    static constexpr void ger(const Layout layout, const_dif_t m, const_dif_t n,
                              const Alpha alpha, const X *x, const_dif_t incx,
                              const Y *y, const_dif_t incy, A *a,
                              const_dif_t lda) {
        ensure_no_null_pointer_(a, x, y);
        ensure_not_negative_<const int>(m, n, lda, incx, incy);
        if constexpr (is_Same_Ty<float, A, X, Y>) {
            cublasSger_v2(
                plat::get_active_device().cublas_handle(),
                static_cast<const int>(m), static_cast<const int>(n),
                static_cast<const X>(alpha), x, static_cast<const int>(incx), y,
                static_cast<const int>(incy), a, static_cast<const int>(lda));
            return;
        }
        if constexpr (is_Same_Ty<double, A, X, Y>) {
            cublasDger_v2(
                plat::get_active_device().cublas_handle(),
                static_cast<const int>(m), static_cast<const int>(n),
                static_cast<const X>(alpha), x, static_cast<const int>(incx), y,
                static_cast<const int>(incy), a, static_cast<const int>(lda));
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief c = alpha*op(a)*op(b) +bata*c
     * @remarks GPU 版本
     */
    template <is_Arithmetic A, is_Arithmetic B, is_Arithmetic C,
              is_Arithmetic Alpha, is_Arithmetic Beta>
    static constexpr void
    gemm(const Layout layout, const Transpose trans_a, const Transpose trans_b,
         const_dif_t m, const_dif_t n, const_dif_t k, const Alpha alpha,
         const A *a, const_dif_t lda, const B *b, const_dif_t ldb,
         const Beta beta, C *c, const_dif_t ldc) {
        ensure_no_null_pointer_(a, b, c);
        ensure_not_negative_<const int>(m, n, k, lda, ldb, ldc);
        if constexpr (is_Same_Ty<float, A, B, C>) {
            cublasSgemm_v2(
                plat::get_active_device().cublas_handle(), convert_(trans_a),
                convert_(trans_b), static_cast<const int>(m),
                static_cast<const int>(n), static_cast<const int>(k),
                static_cast<const A>(alpha), a, static_cast<const int>(lda), b,
                static_cast<const int>(ldb), static_cast<const X>(beta), c,
                static_cast<const int>(ldc));
            return;
        }
        if constexpr (is_Same_Ty<double, A, B, C>) {
            cublasDgemm_v2(
                plat::get_active_device().cublas_handle(), convert_(trans_a),
                convert_(trans_b), static_cast<const int>(m),
                static_cast<const int>(n), static_cast<const int>(k),
                static_cast<const A>(alpha), a, static_cast<const int>(lda), b,
                static_cast<const int>(ldb), static_cast<const X>(beta), c,
                static_cast<const int>(ldc));
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };
};
} // namespace llframe::blas
#endif //__LLFRAME_BLAS_ADAPTER_GPU_HPP__