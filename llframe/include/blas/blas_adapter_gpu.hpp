//    Copyright 2023 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expres__s or implied.
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
    using features = Blas_Adapter_Features<device::GPU>;

public:
    using size_type = typename features::size_type;
    using difference_type = typename features::difference_type;
    using const_dif_t = typename features::const_dif_t;
    using device_type = typename features::device_type;

    using Layout = typename features::Layout;
    using Transpose = typename features::Transpose;
    using Uplo = typename features::Uplo;
    using Diag = typename features::Diag;
    using Side = typename features::Side;

    using plat = typename features::plat;

protected:
    // 根据transpose 和 layout 调整 transpose
    static const cublasOperation_t convert_(const Transpose trans,
                                            const Layout layout) {
        if (layout == Layout::Col_Major) {
            if (trans == Transpose::NoTrans)
                return cublasOperation_t::CUBLAS_OP_N;
            if (trans == Transpose::Trans)
                return cublasOperation_t::CUBLAS_OP_T;
        } else if (layout == Layout::Row_Major) {
            if (trans == Transpose::NoTrans)
                return cublasOperation_t::CUBLAS_OP_T;
            if (trans == Transpose::Trans)
                return cublasOperation_t::CUBLAS_OP_N;
        }
        __LLFRAME_THROW_EXCEPTION_INFO__(exception::Bad_Parameter,
                                         "cant not convert transpose!")
    }

    static const cublasOperation_t convert_(const Transpose trans) {
        if (trans == Transpose::NoTrans) return cublasOperation_t::CUBLAS_OP_N;
        if (trans == Transpose::Trans) return cublasOperation_t::CUBLAS_OP_T;

        __LLFRAME_THROW_EXCEPTION_INFO__(exception::Bad_Parameter,
                                         "cant not convert transpose!")
    }

public:
    /**
     * @brief 向量x的绝对值和
     * @remarks GPU 版本
     */
    template <is_Arithmetic X>
    static constexpr X asum(const_dif_t n, const X *x, const_dif_t incx) {
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            X res__{};
            cublasSasum_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), x,
                           static_cast<const int>(incx), &res__);
            return res__;
        }
        if constexpr (is_Same_Ty<double, X>) {
            X res__{};
            cublasDasum_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), x,
                           static_cast<const int>(incx), &res__);
            return res__;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief res = sum(xi*yi)
     *  @remarks GPU 版本
     */
    template <is_Arithmetic X, is_Arithmetic Y>
    static constexpr X dot(const_dif_t n, const X *x, const_dif_t incx,
                           const Y *y, const_dif_t incy) {
        ensure_no_null_pointer_(x, y);
        ensure_not_negative_<const int>(n, incx, incy);
        if constexpr (is_Same_Ty<float, X, Y>) {
            X res__{};
            cublasSdot_v2(plat::get_active_device().cublas_handle(),
                          static_cast<const int>(n), x,
                          static_cast<const int>(incx), y,
                          static_cast<const int>(incy), &res__);
            return res__;
        }
        if constexpr (is_Same_Ty<double, X, Y>) {
            X res__{};
            cublasDdot_v2(plat::get_active_device().cublas_handle(),
                          static_cast<const int>(n), x,
                          static_cast<const int>(incx), y,
                          static_cast<const int>(incy), &res__);
            return res__;
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
            X res__{};
            cublasSnrm2_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), x,
                           static_cast<const int>(incx), &res__);
            return res__;
        }
        if constexpr (is_Same_Ty<double, X>) {
            X res__{};
            cublasDnrm2_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), x,
                           static_cast<const int>(incx), &res__);
            return res__;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 绝对值最大的第一个索引
     * @remarks GPU 版本
     * @note cublas返回值从1开始
     */
    template <is_Arithmetic X>
    static constexpr difference_type iamax(const_dif_t n, const X *x,
                                           const_dif_t incx) {
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            int res__{};
            cublasIsamax_v2(plat::get_active_device().cublas_handle(),
                            static_cast<const int>(n), x,
                            static_cast<const int>(incx), &res__);
            return static_cast<difference_type>(res__) - 1;
        }
        if constexpr (is_Same_Ty<double, X>) {
            int res__{};
            cublasIdamax_v2(plat::get_active_device().cublas_handle(),
                            static_cast<const int>(n), x,
                            static_cast<const int>(incx), &res__);
            return static_cast<difference_type>(res__) - 1;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 绝对值最小的第一个索引
     * @remarks GPU 版本
     * @note cublas返回值从1开始
     */
    template <is_Arithmetic X>
    static constexpr difference_type iamin(const_dif_t n, const X *x,
                                           const_dif_t incx) {
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            int res__{};
            cublasIsamin_v2(plat::get_active_device().cublas_handle(),
                            static_cast<const int>(n), x,
                            static_cast<const int>(incx), &res__);
            return static_cast<difference_type>(res__) - 1;
        }
        if constexpr (is_Same_Ty<double, X>) {
            int res__{};
            cublasIdamin_v2(plat::get_active_device().cublas_handle(),
                            static_cast<const int>(n), x,
                            static_cast<const int>(incx), &res__);
            return static_cast<difference_type>(res__) - 1;
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
            const X alpha__ = static_cast<const X>(alpha);
            cublasSaxpy_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), &alpha__, x,
                           static_cast<const int>(incx), y,
                           static_cast<const int>(incy));
            return;
        }
        if constexpr (is_Same_Ty<double, X, Y>) {
            const X alpha__ = static_cast<const X>(alpha);
            cublasDaxpy_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), &alpha__, x,
                           static_cast<const int>(incx), y,
                           static_cast<const int>(incy));
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
            cublasScopy_v2(plat::get_active_device().cublas_handle(),
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
            cublasSswap_v2(plat::get_active_device().cublas_handle(),
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
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            const X alpha__ = static_cast<const X>(alpha);
            cublasSscal_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), &alpha__, x,
                           static_cast<const int>(incx));
            return;
        }
        if constexpr (is_Same_Ty<double, X>) {
            const X alpha__ = static_cast<const X>(alpha);
            cublasDscal_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), &alpha__, x,
                           static_cast<const int>(incx));
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief y = alpha*op(a)*x + beta*y;
     * op(a)->m*n;
     * x->1*n(noTrans)/1*m(Trans);
     * y->1*m(noTrans)/1*n(Trans)
     *
     */
    template <is_Arithmetic X, is_Arithmetic Y, is_Arithmetic A,
              is_Arithmetic Alpha, is_Arithmetic Beta>
    static constexpr void gemv(const Layout layout, const Transpose trans,
                               const_dif_t m, const_dif_t n, const Alpha alpha,
                               const A *a, const_dif_t lda, const X *x,
                               const_dif_t incx, const Beta beta, Y *y,
                               const_dif_t incy) {
        ensure_no_null_pointer_(a, x, y);
        ensure_not_negative_<const int>(m, n, lda, incx, incy);
        ensure_ld_legal_(layout, m, n, lda);
        if constexpr (is_Same_Ty<float, A, X, Y>) {
            const X alpha___ = static_cast<const X>(alpha);
            const X beta__ = static_cast<const X>(beta);
            if (layout == Layout::Row_Major) {
                cublasSgemv_v2(plat::get_active_device().cublas_handle(),
                               convert_(trans, layout),
                               static_cast<const int>(n),
                               static_cast<const int>(m), &alpha___, a,
                               static_cast<const int>(lda), x,
                               static_cast<const int>(incx), &beta__, y,
                               static_cast<const int>(incy));
            } else if (layout == Layout::Col_Major) {
                cublasSgemv_v2(plat::get_active_device().cublas_handle(),
                               convert_(trans, layout),
                               static_cast<const int>(m),
                               static_cast<const int>(n), &alpha___, a,
                               static_cast<const int>(lda), x,
                               static_cast<const int>(incx), &beta__, y,
                               static_cast<const int>(incy));
            }
            return;
        }
        if constexpr (is_Same_Ty<double, A, X, Y>) {
            const X alpha___ = static_cast<const X>(alpha);
            const X beta__ = static_cast<const X>(beta);
            if (layout == Layout::Row_Major) {
                cublasDgemv_v2(plat::get_active_device().cublas_handle(),
                               convert_(trans, layout),
                               static_cast<const int>(n),
                               static_cast<const int>(m), &alpha___, a,
                               static_cast<const int>(lda), x,
                               static_cast<const int>(incx), &beta__, y,
                               static_cast<const int>(incy));
            } else if (layout == Layout::Col_Major) {
                cublasDgemv_v2(plat::get_active_device().cublas_handle(),
                               convert_(trans, layout),
                               static_cast<const int>(m),
                               static_cast<const int>(n), &alpha___, a,
                               static_cast<const int>(lda), x,
                               static_cast<const int>(incx), &beta__, y,
                               static_cast<const int>(incy));
            }
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
        ensure_ld_legal_(layout, m, n, lda);
        if constexpr (is_Same_Ty<float, A, X, Y>) {
            const A alpha__ = static_cast<const A>(alpha);
            if (layout == Layout::Row_Major) {
                cublasSger_v2(plat::get_active_device().cublas_handle(),
                              static_cast<const int>(n),
                              static_cast<const int>(m), &alpha__, y,
                              static_cast<const int>(incy), x,
                              static_cast<const int>(incx), a,
                              static_cast<const int>(lda));
            } else {
                cublasSger_v2(plat::get_active_device().cublas_handle(),
                              static_cast<const int>(m),
                              static_cast<const int>(n), &alpha__, x,
                              static_cast<const int>(incx), y,
                              static_cast<const int>(incy), a,
                              static_cast<const int>(lda));
            }
            return;
        }
        if constexpr (is_Same_Ty<double, A, X, Y>) {
            const A alpha__ = static_cast<const A>(alpha);
            if (layout == Layout::Row_Major) {
                cublasDger_v2(plat::get_active_device().cublas_handle(),
                              static_cast<const int>(n),
                              static_cast<const int>(m), &alpha__, y,
                              static_cast<const int>(incy), x,
                              static_cast<const int>(incx), a,
                              static_cast<const int>(lda));
            } else {
                cublasDger_v2(plat::get_active_device().cublas_handle(),
                              static_cast<const int>(m),
                              static_cast<const int>(n), &alpha__, x,
                              static_cast<const int>(incx), y,
                              static_cast<const int>(incy), a,
                              static_cast<const int>(lda));
            }
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
        ensure_ld_legal_(layout, trans_a, m, k, lda);
        ensure_ld_legal_(layout, trans_b, k, n, ldb);
        ensure_ld_legal_(layout, m, n, ldc);
        if constexpr (is_Same_Ty<float, A, B, C>) {
            const A alpha__ = static_cast<const A>(alpha);
            const A beta__ = static_cast<const A>(beta);
            if (layout == Layout::Row_Major) {
                cublasSgemm_v2(
                    plat::get_active_device().cublas_handle(),
                    convert_(trans_a), convert_(trans_b),
                    static_cast<const int>(k), static_cast<const int>(m),
                    static_cast<const int>(k), &alpha__, b,
                    static_cast<const int>(k), a, static_cast<const int>(k),
                    &beta__, c, static_cast<const int>(n));
            } else {
                cublasSgemm_v2(
                    plat::get_active_device().cublas_handle(),
                    convert_(trans_a, layout), convert_(trans_b, layout),
                    static_cast<const int>(m), static_cast<const int>(n),
                    static_cast<const int>(k), &alpha__, a,
                    static_cast<const int>(lda), b, static_cast<const int>(ldb),
                    &beta__, c, static_cast<const int>(ldc));
            }
            return;
        }
        if constexpr (is_Same_Ty<double, A, B, C>) {
            const A alpha__ = static_cast<const A>(alpha);
            const A beta__ = static_cast<const A>(beta);
            if (layout == Layout::Row_Major) {
                cublasDgemm_v2(
                    plat::get_active_device().cublas_handle(),
                    convert_(trans_a), convert_(trans_b),
                    static_cast<const int>(k), static_cast<const int>(m),
                    static_cast<const int>(k), &alpha__, b,
                    static_cast<const int>(k), a, static_cast<const int>(k),
                    &beta__, c, static_cast<const int>(n));
            } else {
                cublasDgemm_v2(
                    plat::get_active_device().cublas_handle(),
                    convert_(trans_a, layout), convert_(trans_b, layout),
                    static_cast<const int>(m), static_cast<const int>(n),
                    static_cast<const int>(k), &alpha__, a,
                    static_cast<const int>(lda), b, static_cast<const int>(ldb),
                    &beta__, c, static_cast<const int>(ldc));
            }
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };
};
} // namespace llframe::blas
#endif //__LLFRAME_BLAS_ADAPTER_GPU_HPP__