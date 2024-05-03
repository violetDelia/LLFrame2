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
 * @remarks CPU 版本
 */
#ifndef __LLFRAME_BLAS_ADAPTER_CPU_HPP__
#define __LLFRAME_BLAS_ADAPTER_CPU_HPP__
#include "blas/blas_adapter.hpp"
#include "device/device_impl.hpp"
#include "openblas/cblas.h"
namespace llframe::blas {
template <>
class Blas_Adapter<device::CPU> : public _Blas_Adapter_Base<device::CPU> {
public:
    using Self = Blas_Adapter<device::CPU>;
    using Base = _Blas_Adapter_Base<device::CPU>;

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
    // openblas的bug,调用axpy最好不要超过该值
    static constexpr size_type openblas_max_axpy_n = 8192;
    // openblas的bug,调用scal最好不要超过该值
    static constexpr size_type openblas_max_scal_n = 8192;

protected:
    static const CBLAS_LAYOUT convert_(const Layout layout) {
        if (layout == Layout::Col_Major) return CBLAS_LAYOUT::CblasColMajor;
        if (layout == Layout::Row_Major) return CBLAS_LAYOUT::CblasRowMajor;
        __LLFRAME_THROW_EXCEPTION_INFO__(exception::Bad_Parameter,
                                         "cant not convert layout!")
    }

    static const CBLAS_TRANSPOSE convert_(const Transpose trans) {
        if (trans == Transpose::NoTrans) return CBLAS_TRANSPOSE::CblasNoTrans;
        if (trans == Transpose::Trans) return CBLAS_TRANSPOSE::CblasTrans;
        if (trans == Transpose::ConjNoTrans)
            return CBLAS_TRANSPOSE::CblasConjNoTrans;
        if (trans == Transpose::ConjTrans)
            return CBLAS_TRANSPOSE::CblasConjTrans;
        __LLFRAME_THROW_EXCEPTION_INFO__(exception::Bad_Parameter,
                                         "cant not convert transpose!")
    }

public:
    /**
     * @brief 向量x的绝对值和
     * @remarks CPU 版本
     */
    template <is_Arithmetic X>
    static constexpr X asum(const_dif_t n, const X *x, const_dif_t incx) {
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            return cblas_sasum(static_cast<const int>(n), x,
                               static_cast<const int>(incx));
        }
        if constexpr (is_Same_Ty<double, X>) {
            return cblas_dasum(static_cast<const int>(n), x,
                               static_cast<const int>(incx));
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 向量x的和
     * @remarks CPU 版本
     */
    template <is_Arithmetic X>
    static constexpr X sum(const_dif_t n, const X *x, const_dif_t incx) {
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            return cblas_ssum(static_cast<const int>(n), x,
                              static_cast<const int>(incx));
        }
        if constexpr (is_Same_Ty<double, X>) {
            return cblas_dsum(static_cast<const int>(n), x,
                              static_cast<const int>(incx));
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief res = sum(xi*yi)
     * @remarks CPU 版本
     */
    template <is_Arithmetic X, is_Arithmetic Y>
    static constexpr X dot(const_dif_t n, const X *x, const_dif_t incx,
                           const Y *y, const_dif_t incy) {
        ensure_no_null_pointer_(x, y);
        ensure_not_negative_<const int>(n, incx, incy);
        if constexpr (is_Same_Ty<float, X, Y>) {
            return cblas_sdot(static_cast<const int>(n), x,
                              static_cast<const int>(incx), y,
                              static_cast<const int>(incy));
        }
        if constexpr (is_Same_Ty<double, X, Y>) {
            return cblas_ddot(static_cast<const int>(n), x,
                              static_cast<const int>(incx), y,
                              static_cast<const int>(incy));
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief res = ||x||^2
     * @remarks CPU 版本
     */
    template <is_Arithmetic X>
    static constexpr X nrm2(const_dif_t n, const X *x, const_dif_t incx) {
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            return cblas_snrm2(static_cast<const int>(n), x,
                               static_cast<const int>(incx));
        }
        if constexpr (is_Same_Ty<double, X>) {
            return cblas_dnrm2(static_cast<const int>(n), x,
                               static_cast<const int>(incx));
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 绝对值最大的第一个索引
     * @remarks CPU 版本
     */
    template <is_Arithmetic X>
    static constexpr difference_type iamax(const_dif_t n, const X *x,
                                           const_dif_t incx) {
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            return cblas_isamax(static_cast<const int>(n), x,
                                static_cast<const int>(incx));
        }
        if constexpr (is_Same_Ty<double, X>) {
            return cblas_idamax(static_cast<const int>(n), x,
                                static_cast<const int>(incx));
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 绝对值最小的第一个索引
     * @remarks CPU 版本
     */
    template <is_Arithmetic X>
    static constexpr difference_type iamin(const_dif_t n, const X *x,
                                           const_dif_t incx) {
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            return cblas_isamin(static_cast<const int>(n), x,
                                static_cast<const int>(incx));
        }
        if constexpr (is_Same_Ty<double, X>) {
            return cblas_idamin(static_cast<const int>(n), x,
                                static_cast<const int>(incx));
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief x中第一个最大值的索引
     * @remarks CPU 版本
     * @note 仅openblas支持
     */
    template <is_Arithmetic X>
    static constexpr difference_type imax(const_dif_t n, const X *x,
                                          const_dif_t incx) {
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            return cblas_ismax(static_cast<const int>(n), x,
                               static_cast<const int>(incx));
        }
        if constexpr (is_Same_Ty<double, X>) {
            return cblas_idmax(static_cast<const int>(n), x,
                               static_cast<const int>(incx));
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief x中第一个最小值的索引
     * @remarks CPU 版本
     * @note 仅openblas支持
     */
    template <is_Arithmetic X>
    static constexpr difference_type imin(const_dif_t n, const X *x,
                                          const_dif_t incx) {
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            return cblas_ismin(static_cast<const int>(n), x,
                               static_cast<const int>(incx));
        }
        if constexpr (is_Same_Ty<double, X>) {
            return cblas_idmin(static_cast<const int>(n), x,
                               static_cast<const int>(incx));
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief y = alpha*x+y
     * @remarks CPU 版本
     */
    template <is_Arithmetic X, is_Arithmetic Y, is_Arithmetic Alpha>
    static constexpr void axpy(const_dif_t n, const Alpha alpha, const X *x,
                               const_dif_t incx, Y *y, const_dif_t incy) {
        ensure_no_null_pointer_(x, y);
        ensure_not_negative_<const int>(n, incx, incy);
        if constexpr (is_Same_Ty<float, X, Y>) {
            size_t count = static_cast<int>(n);
            while (count >= openblas_max_axpy_n) {
                cblas_saxpy(static_cast<const int>(openblas_max_axpy_n),
                            static_cast<const X>(alpha), x,
                            static_cast<const int>(incx), y,
                            static_cast<const int>(incy));
                count -= openblas_max_axpy_n;
                x += openblas_max_axpy_n * static_cast<const int>(incx);
                y += openblas_max_axpy_n * static_cast<const int>(incy);
            }
            if (count) {
                cblas_saxpy(static_cast<const int>(count),
                            static_cast<const X>(alpha), x,
                            static_cast<const int>(incx), y,
                            static_cast<const int>(incy));
            }
            return;
        }
        if constexpr (is_Same_Ty<double, X, Y>) {
            size_t count = static_cast<int>(n);
            while (count >= openblas_max_axpy_n) {
                cblas_daxpy(static_cast<const int>(openblas_max_axpy_n),
                            static_cast<const X>(alpha), x,
                            static_cast<const int>(incx), y,
                            static_cast<const int>(incy));
                count -= openblas_max_axpy_n;
                x += openblas_max_axpy_n * static_cast<const int>(incx);
                y += openblas_max_axpy_n * static_cast<const int>(incy);
            }
            if (count) {
                cblas_daxpy(static_cast<const int>(count),
                            static_cast<const X>(alpha), x,
                            static_cast<const int>(incx), y,
                            static_cast<const int>(incy));
            }
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief y = x
     * @remarks CPU 版本
     */
    template <is_Arithmetic X, is_Arithmetic Y>
    static constexpr void copy(const_dif_t n, const X *x, const_dif_t incx,
                               Y *y, const_dif_t incy) {
        ensure_no_null_pointer_(x, y);
        ensure_not_negative_<const int>(n, incx, incy);
        if constexpr (is_Same_Ty<float, X, Y>) {
            cblas_scopy(static_cast<const int>(n), x,
                        static_cast<const int>(incx), y,
                        static_cast<const int>(incy));
            return;
        }
        if constexpr (is_Same_Ty<double, X, Y>) {
            cblas_dcopy(static_cast<const int>(n), x,
                        static_cast<const int>(incx), y,
                        static_cast<const int>(incy));
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief y = x;x = y
     * @remarks CPU 版本
     */
    template <is_Arithmetic X, is_Arithmetic Y>
    static constexpr void swap(const_dif_t n, X *x, const_dif_t incx, Y *y,
                               const_dif_t incy) {
        ensure_no_null_pointer_(x, y);
        ensure_not_negative_<const int>(n, incx, incy);
        if constexpr (is_Same_Ty<float, X, Y>) {
            cblas_sswap(static_cast<const int>(n), x,
                        static_cast<const int>(incx), y,
                        static_cast<const int>(incy));
            return;
        }
        if constexpr (is_Same_Ty<double, X, Y>) {
            cblas_dswap(static_cast<const int>(n), x,
                        static_cast<const int>(incx), y,
                        static_cast<const int>(incy));
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief x = alpha * x
     * @remarks CPU 版本
     */
    template <is_Arithmetic X, is_Arithmetic Alpha>
    static constexpr void scal(const_dif_t n, const Alpha alpha, X *x,
                               const_dif_t incx) {
        ensure_no_null_pointer_(x);
        ensure_not_negative_<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            size_t count = static_cast<int>(n);
            while (count >= openblas_max_scal_n) {
                cblas_sscal(static_cast<const int>(openblas_max_scal_n),
                            static_cast<const X>(alpha), x,
                            static_cast<const int>(incx));
                count -= openblas_max_axpy_n;
                x += openblas_max_axpy_n * static_cast<const int>(incx);
            }
            if (count) {
                cblas_sscal(static_cast<const int>(count),
                            static_cast<const X>(alpha), x,
                            static_cast<const int>(incx));
            }
            return;
        }
        if constexpr (is_Same_Ty<double, X>) {
            size_t count = static_cast<int>(n);
            while (count >= openblas_max_scal_n) {
                cblas_dscal(static_cast<const int>(openblas_max_scal_n),
                            static_cast<const X>(alpha), x,
                            static_cast<const int>(incx));
                count -= openblas_max_axpy_n;
                x += openblas_max_axpy_n * static_cast<const int>(incx);
            }
            if (count) {
                cblas_dscal(static_cast<const int>(count),
                            static_cast<const X>(alpha), x,
                            static_cast<const int>(incx));
            }
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief y = alpha*op(a)*x + beta*y;
     * @remarks CPU 版本 op(a)->m*n;
     * x->1*n(noTrans)/1*m(Trans);
     * y->1*m(noTrans)/1*n(Trans)
     *@note 内存大于2^28比特时会出错
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
            cblas_sgemv(
                convert_(layout), convert_(trans), static_cast<const int>(m),
                static_cast<const int>(n), static_cast<const X>(alpha), a,
                static_cast<const int>(lda), x, static_cast<const int>(incx),
                static_cast<const X>(beta), y, static_cast<const int>(incy));
            return;
        }
        if constexpr (is_Same_Ty<double, A, X, Y>) {
            cblas_dgemv(
                convert_(layout), convert_(trans), static_cast<const int>(m),
                static_cast<const int>(n), static_cast<const X>(alpha), a,
                static_cast<const int>(lda), x, static_cast<const int>(incx),
                static_cast<const X>(beta), y, static_cast<const int>(incy));
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief a = alpha*x*y^T + a
     * @remarks CPU 版本
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
            cblas_sger(convert_(layout), static_cast<const int>(m),
                       static_cast<const int>(n), static_cast<const X>(alpha),
                       x, static_cast<const int>(incx), y,
                       static_cast<const int>(incy), a,
                       static_cast<const int>(lda));
            return;
        }
        if constexpr (is_Same_Ty<double, A, X, Y>) {
            cblas_dger(convert_(layout), static_cast<const int>(m),
                       static_cast<const int>(n), static_cast<const X>(alpha),
                       x, static_cast<const int>(incx), y,
                       static_cast<const int>(incy), a,
                       static_cast<const int>(lda));
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief c = alpha*op(a)*op(b) +bata*c
     * @remarks CPU 版本
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
            cblas_sgemm(convert_(layout), convert_(trans_a), convert_(trans_b),
                        static_cast<const int>(m), static_cast<const int>(n),
                        static_cast<const int>(k), static_cast<const A>(alpha),
                        a, static_cast<const int>(lda), b,
                        static_cast<const int>(ldb), static_cast<const A>(beta),
                        c, static_cast<const int>(ldc));
            return;
        }
        if constexpr (is_Same_Ty<double, A, B, C>) {
            cblas_dgemm(convert_(layout), convert_(trans_a), convert_(trans_b),
                        static_cast<const int>(m), static_cast<const int>(n),
                        static_cast<const int>(k), static_cast<const A>(alpha),
                        a, static_cast<const int>(lda), b,
                        static_cast<const int>(ldb), static_cast<const A>(beta),
                        c, static_cast<const int>(ldc));
            return;
        }
        __THROW_UNIMPLEMENTED__;
    };
};
} // namespace llframe::blas
#endif //__LLFRAME_BLAS_ADAPTER_CPU_HPP__