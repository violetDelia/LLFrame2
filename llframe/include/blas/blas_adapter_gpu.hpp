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

public:
    /**
     * @brief 向量x的绝对值和
     */
    template <is_Arithmetic X>
    static constexpr X asum(const_dif_t n, const X *x, const_dif_t incx) {
        ensure_no_null_pointer(x);
        ensure_not_negative<const int>(n, incx);
        if constexpr (is_Same_Ty<float, X>) {
            X sum{};
            cublasSasum_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), x,
                           static_cast<const int>(incx), &sum);
            return sum;
        }
        if constexpr (is_Same_Ty<double, X>) {
            X sum{};
            cublasDasum_v2(plat::get_active_device().cublas_handle(),
                           static_cast<const int>(n), x,
                           static_cast<const int>(incx), &sum);
            return sum;
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
            X sum{};
            cublasSdot_v2(plat::get_active_device().cublas_handle(),
                          static_cast<const int>(n), x,
                          static_cast<const int>(incx), y,
                          static_cast<const int>(incy), &sum);
            return sum;
        }
        if constexpr (is_Same_Ty<double, X>) {
            X sum{};
            cublasDdot_v2(plat::get_active_device().cublas_handle(),
                          static_cast<const int>(n), x,
                          static_cast<const int>(incx), y,
                          static_cast<const int>(incy), &sum);
            return sum;
        }

        __THROW_UNIMPLEMENTED__;
    };
};
} // namespace llframe::blas
#endif //__LLFRAME_BLAS_ADAPTER_GPU_HPP__