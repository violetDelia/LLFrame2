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

public:
    /**
     * @brief 向量x的绝对值和
     */
    template <is_Arithmetic X>
    static constexpr X asum(const_dif_t n, const X *x, const_dif_t incx) {
        ensure_no_null_pointer(x);
        ensure_not_negative<const int>(n, incx);
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
     */
    template <is_Arithmetic X>
    static constexpr X sum(const_dif_t n, const X *x, const_dif_t incx) {
        ensure_no_null_pointer(x);
        ensure_not_negative<const int>(n, incx);
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
     */
    template <is_Arithmetic X, is_Arithmetic Y>
    static constexpr X dot(const_dif_t n, const X *x, const_dif_t incx,
                           const Y *y, const_dif_t incy) {
        ensure_no_null_pointer(x, y);
        ensure_not_negative<const int>(n, incx, incy);
        if constexpr (is_Same_Ty<float, X>) {
            return cblas_sdot(static_cast<const int>(n), x,
                              static_cast<const int>(incx), y,
                              static_cast<const int>(incy));
        }
        if constexpr (is_Same_Ty<double, X>) {
            return cblas_ddot(static_cast<const int>(n), x,
                              static_cast<const int>(incx), y,
                              static_cast<const int>(incy));
        }
        __THROW_UNIMPLEMENTED__;
    };
};
} // namespace llframe::blas
#endif //__LLFRAME_BLAS_ADAPTER_CPU_HPP__