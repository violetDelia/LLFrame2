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
 *
 */
#ifndef LLFRAME_BLAS_BLAS_ADAPTER_HPP
#define LLFRAME_BLAS_BLAS_ADAPTER_HPP
#include "core/exception.hpp"
#include "core/base_type.hpp"
#include "device/device_platform.hpp"
namespace llframe::blas {

template <device::is_Device Device>
struct Blas_Adapter_Features {
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using const_dif_t = const ptrdiff_t;
    using device_type = Device;

    using Layout = Blas_Layout;
    using Transpose = Blas_Transpose;
    using Uplo = Blas_Uplo;
    using Diag = Blas_Diag;
    using Side = Blas_Side;

    using plat = device::Device_Platform<Device>;
};

// blas_adapter的一般功能实现
template <device::is_Device Device>
class _Blas_Adapter_Base {
private:
    using Self = _Blas_Adapter_Base<Device>;
    using features = Blas_Adapter_Features<Device>;

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

protected: // 参数检查辅助方法
    // 判断真正是否为空
    template <is_Pointer... Pointers>
    static constexpr void ensure_no_null_pointer_(const Pointers... pointers) {
        if constexpr (sizeof...(Pointers) == 0) return;
        if ((... || (pointers == nullptr))) {
            __LLFRAME_THROW_EXCEPTION_INFO__(exception::Null_Pointer,
                                             "has null pointer!")
        }
    }

    // 判断数值被转换后是否为负数
    template <class Trans_Ty, is_Arithmetic... Arithmetics>
    static constexpr void
    ensure_not_negative_(const Arithmetics... arithmetics) {
        if constexpr (sizeof...(Arithmetics) == 0) return;
        if ((... || (static_cast<Trans_Ty>(arithmetics) < 0))) {
            __LLFRAME_THROW_EXCEPTION_INFO__(
                exception::Bad_Parameter,
                "parameters is negative after convert!")
        }
    }

    // 确保ld参数是合法的
    static constexpr void ensure_ld_legal_(const Layout layout, const_dif_t m,
                                           const_dif_t n, const_dif_t ld) {
        if (((layout == Layout::Row_Major) && (ld >= n))
            || ((layout == Layout::Col_Major) && (ld >= m))) {
            return;
        }
        __LLFRAME_THROW_EXCEPTION_INFO__(exception::Bad_Parameter,
                                         "leading demention is illegal!")
    };

    // 确保ld参数是合法的
    static constexpr void ensure_ld_legal_(const Layout layout,
                                           const Transpose transpose,
                                           const_dif_t m, const_dif_t n,
                                           const_dif_t ld) {
        if (transpose == Transpose::NoTrans) {
            if (((layout == Layout::Row_Major) && (ld >= n))
                || ((layout == Layout::Col_Major) && (ld >= m))) {
                return;
            }
        }
        if (transpose == Transpose::Trans) {
            if (((layout == Layout::Row_Major) && (ld >= m))
                || ((layout == Layout::Col_Major) && (ld >= n))) {
                return;
            }
        }
        __LLFRAME_THROW_EXCEPTION_INFO__(exception::Bad_Parameter,
                                         "leading demention is illegal!")
    };

public: //
    static constexpr void set_num_threads(int num_threads) {
        __THROW_UNIMPLEMENTED__;
    };

    static constexpr int openblas_get_num_threads(void) {
        __THROW_UNIMPLEMENTED__;
    };

    static constexpr int openblas_get_num_procs(void) {
        __THROW_UNIMPLEMENTED__;
    };

    static constexpr char *openblas_get_config(void) {
        __THROW_UNIMPLEMENTED__;
    };

    static constexpr char *openblas_get_corename(void) {
        __THROW_UNIMPLEMENTED__;
    };

public:
    /**
     * @brief res = sum(xi*yi)
     */
    template <is_Arithmetic X, is_Arithmetic Y>
    static constexpr X dot(const_dif_t n, const X *x, const_dif_t incx,
                           const Y *y, const_dif_t incy, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 向量x的绝对值和
     */
    template <is_Arithmetic X>
    static constexpr X asum(const_dif_t n, const X *x, const_dif_t incx,
                            const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 向量x的和
     *
     * @note 仅openblas支持
     */
    template <is_Arithmetic X>
    static constexpr X sum(const_dif_t n, const X *x, const_dif_t incx,
                           const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief res = ||x||^2
     *
     */
    template <is_Arithmetic X>
    static constexpr X nrm2(const_dif_t n, const X *x, const_dif_t incx,
                            const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 绝对值最大的第一个索引
     *
     */
    template <is_Arithmetic X>
    static constexpr difference_type
    iamax(const_dif_t n, const X *x, const_dif_t incx, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 绝对值最小的第一个索引
     *
     */
    template <is_Arithmetic X>
    static constexpr difference_type
    iamin(const_dif_t n, const X *x, const_dif_t incx, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief x中第一个最大值的索引
     *
     * @note 仅openblas支持
     */
    template <is_Arithmetic X>
    static constexpr difference_type
    imax(const_dif_t n, const X *x, const_dif_t incx, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief x中第一个最小值的索引
     *
     * @note 仅openblas支持
     */
    template <is_Arithmetic X>
    static constexpr difference_type
    imin(const_dif_t n, const X *x, const_dif_t incx, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief y = alpha*x+y
     *
     */
    template <is_Arithmetic X, is_Arithmetic Y, is_Arithmetic Alpha>
    static constexpr void axpy(const_dif_t n, const Alpha alpha, const X *x,
                               const_dif_t incx, Y *y, const_dif_t incy,
                               const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief y = x
     *
     */
    template <is_Arithmetic X, is_Arithmetic Y>
    static constexpr void copy(const_dif_t n, const X *x, const_dif_t incx,
                               Y *y, const_dif_t incy, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief y = x;x = y
     *
     */
    template <is_Arithmetic X, is_Arithmetic Y>
    static constexpr void swap(const_dif_t n, X *x, const_dif_t incx, Y *y,
                               const_dif_t incy, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic X, is_Arithmetic Y, is_Arithmetic C,
              is_Arithmetic S>
    static constexpr void rot(const_dif_t n, X *x, const_dif_t incx, Y *y,
                              const_dif_t incy, const C c, const S s,
                              const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic A, is_Arithmetic B, is_Arithmetic C,
              is_Arithmetic S>
    static constexpr void rotg(A *a, B *b, C *c, S *s, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic X, is_Arithmetic Y, is_Arithmetic P>
    static constexpr void rotm(const_dif_t n, X *x, const_dif_t incx, Y *y,
                               const_dif_t incy, const P *p,
                               const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic D1, is_Arithmetic D2, is_Arithmetic B1,
              is_Arithmetic B2, is_Arithmetic P>
    static constexpr void rotmg(D1 *d1, D2 *d2, B1 *b1, const B2 b2, P *p,
                                const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief x = alpha*x
     *
     */
    template <is_Arithmetic X, is_Arithmetic Alpha>
    static constexpr void scal(const_dif_t n, const Alpha alpha, X *x,
                               const_dif_t incx, const Device &device) {
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
                               const_dif_t incy, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief a = alpha*x*y^T + a
     *
     */
    template <is_Arithmetic X, is_Arithmetic Y, is_Arithmetic A,
              is_Arithmetic Alpha>
    static constexpr void ger(const Layout layout, const_dif_t m, const_dif_t n,
                              const Alpha alpha, const X *x, const_dif_t incx,
                              const Y *y, const_dif_t incy, A *a,
                              const_dif_t lda, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic X, is_Arithmetic A>
    static constexpr void trsv(const Layout layout, const Uplo Uplo,
                               const Transpose trans_a, const Diag Diag,
                               const_dif_t n, const A *a, const_dif_t lda, X *x,
                               const_dif_t incx, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic X, is_Arithmetic A>
    static constexpr void trmv(const Layout layout, const Uplo Uplo,
                               const Transpose trans_a, const Diag Diag,
                               const_dif_t n, const A *a, const_dif_t lda, X *x,
                               const_dif_t incx, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic X, is_Arithmetic A, is_Arithmetic Alpha>
    static constexpr void syr(const Layout layout, const Uplo Uplo,
                              const_dif_t n, const Alpha alpha, const X *x,
                              const_dif_t incx, A *a, const_dif_t lda,
                              const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic X, is_Arithmetic Y, is_Arithmetic A,
              is_Arithmetic Alpha>
    static constexpr void syr2(const Layout layout, const Uplo Uplo,
                               const_dif_t n, const Alpha alpha, const X *x,
                               const_dif_t incx, const Y *y, const_dif_t incy,
                               A *a, const_dif_t lda, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic X, is_Arithmetic Y, is_Arithmetic A,
              is_Arithmetic Alpha, is_Arithmetic Beta>
    static constexpr void
    gbmv(const Layout layout, const Transpose trans_a, const_dif_t m,
         const_dif_t n, const_dif_t kL, const_dif_t kU, const Alpha alpha,
         const A *a, const_dif_t lda, const X *x, const_dif_t incx,
         const Beta beta, Y *y, const_dif_t incy, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic X, is_Arithmetic Y, is_Arithmetic A,
              is_Arithmetic Alpha, is_Arithmetic Beta>
    static constexpr void sbmv(const Layout layout, const Uplo Uplo,
                               const_dif_t n, const_dif_t k, const Alpha alpha,
                               const A *a, const_dif_t lda, const X *x,
                               const_dif_t incx, const Beta beta, Y *y,
                               const_dif_t incy, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic X, is_Arithmetic A>
    static constexpr void
    tbsv(const Layout layout, const Uplo Uplo, const Transpose trans_a,
         const Diag Diag, const_dif_t n, const_dif_t k, const A *a,
         const_dif_t lda, X *x, const_dif_t incx, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic X, is_Arithmetic A>
    static constexpr void tpmv(const Layout layout, const Uplo Uplo,
                               const Transpose trans_a, const Diag Diag,
                               const_dif_t n, const A *ap, X *x,
                               const_dif_t incx, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic X, is_Arithmetic A>
    static constexpr void tpsv(const Layout layout, const Uplo Uplo,
                               const Transpose trans_a, const Diag Diag,
                               const_dif_t n, const A *ap, X *x,
                               const_dif_t incx, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic X, is_Arithmetic Y, is_Arithmetic A,
              is_Arithmetic Alpha, is_Arithmetic Beta>
    static constexpr void
    symv(const Layout layout, const Uplo Uplo, const_dif_t n, const Alpha alpha,
         const A *a, const_dif_t lda, const X *x, const_dif_t incx,
         const Beta beta, Y *y, const_dif_t incy, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic X, is_Arithmetic Y, is_Arithmetic A,
              is_Arithmetic Alpha, is_Arithmetic Beta>
    static constexpr void spmv(const Layout layout, const Uplo Uplo,
                               const_dif_t n, const Alpha alpha, const A *ap,
                               const X *x, const_dif_t incx, const Beta beta,
                               Y *y, const_dif_t incy, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic X, is_Arithmetic A, is_Arithmetic Alpha>
    static constexpr void spr(const Layout layout, const Uplo Uplo,
                              const_dif_t n, const Alpha alpha, const X *x,
                              const_dif_t incx, A *ap, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic X, is_Arithmetic Y, is_Arithmetic A,
              is_Arithmetic Alpha>
    static constexpr void spr2(const Layout layout, const Uplo Uplo,
                               const_dif_t n, const Alpha alpha, const X *x,
                               const_dif_t incx, const Y *y, const_dif_t incy,
                               A *a, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief c = alpha*op(a)*op(b) +bata*c
     *
     */
    template <is_Arithmetic A, is_Arithmetic B, is_Arithmetic C,
              is_Arithmetic Alpha, is_Arithmetic Beta>
    static constexpr void
    gemm(const Layout layout, const Transpose trans_a, const Transpose trans_b,
         const_dif_t m, const_dif_t n, const_dif_t k, const Alpha alpha,
         const A *a, const_dif_t lda, const B *b, const_dif_t ldb,
         const Beta beta, C *c, const_dif_t ldc, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic A, is_Arithmetic B, is_Arithmetic C,
              is_Arithmetic Alpha, is_Arithmetic Beta>
    static constexpr void symm(const Layout layout, const enum IDE Side,
                               const Uplo Uplo, const_dif_t m, const_dif_t n,
                               const Alpha alpha, const A *a, const_dif_t lda,
                               const B *b, const_dif_t ldb, const Beta beta,
                               C *c, const_dif_t ldc, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic A, is_Arithmetic C, is_Arithmetic Alpha,
              is_Arithmetic Beta>
    static constexpr void syrk(const Layout layout, const Uplo Uplo,
                               const Transpose Trans, const_dif_t n,
                               const_dif_t k, const Alpha alpha, const A *a,
                               const_dif_t lda, const Beta beta, C *c,
                               const_dif_t ldc, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic A, is_Arithmetic B, is_Arithmetic C,
              is_Arithmetic Alpha, is_Arithmetic Beta>
    static constexpr void
    syr2k(const Layout layout, const Uplo Uplo, const Transpose Trans,
          const_dif_t n, const_dif_t k, const Alpha alpha, const A *a,
          const_dif_t lda, const B *b, const_dif_t ldb, const Beta beta, C *c,
          const_dif_t ldc, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic A, is_Arithmetic B, is_Arithmetic Alpha>
    static constexpr void trmm(const Layout layout, const enum IDE Side,
                               const Uplo Uplo, const Transpose trans_a,
                               const Diag Diag, const_dif_t m, const_dif_t n,
                               const Alpha alpha, const A *a, const_dif_t lda,
                               B *b, const_dif_t ldb, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 暂时不用
     *
     */
    template <is_Arithmetic A, is_Arithmetic B, is_Arithmetic Alpha>
    static constexpr void trsm(const Layout layout, const enum IDE Side,
                               const Uplo Uplo, const Transpose trans_a,
                               const Diag Diag, const_dif_t m, const_dif_t n,
                               const Alpha alpha, const A *a, const_dif_t lda,
                               B *b, const_dif_t ldb, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    };

public: // openblas extensions
    /**
     * @brief y_i = y_i/x_i;
     *
     */
    template <is_Arithmetic X, is_Arithmetic Y>
    static constexpr void divide_vv(const int n, X *x, const int incx, Y *y,
                                    const int incy, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    }

    /**
     * @brief y_i = y_i*x_i;
     *
     */
    template <is_Arithmetic X, is_Arithmetic Y>
    static constexpr void multiply_vv(const int n, X *x, const int incx, Y *y,
                                      const int incy, const Device &device) {
        __THROW_UNIMPLEMENTED__;
    }
};

/**
 * @brief
 * 根据不同的设备类型选取合适的线性代数函数,为了能够支持混合精度所以
 * 每种指针都当作一个独立的类型去写.后续有需要再进行补充.
 * 后续多线程的改动在Device_Plat上面改,这里面不要动.
 * @note 所有调用都要传入装置,平台操作以及通讯在该类里面实现。
 *
 * @tparam Device
 */
template <device::is_Device Device>
class Blas_Adapter : public _Blas_Adapter_Base<Device> {
public:
    using Self = Blas_Adapter<Device>;
    using Base = _Blas_Adapter_Base<Device>;
    using features = Blas_Adapter_Features<Device>;

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
};

} // namespace llframe::blas
#endif // LLFRAME_BLAS_BLAS_ADAPTER_HPP