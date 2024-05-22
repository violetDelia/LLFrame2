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
 * @brief gpu 算子扩展  vec*vec
 *
 */

#ifndef LLFRAME_BLAS_BLAS_EXTENSION_GPU_MULTIPLY_VV_HPP
#define LLFRAME_BLAS_BLAS_EXTENSION_GPU_MULTIPLY_VV_HPP
#include "core/base_type.hpp"
#include "cuda_runtime.h"
#include "core/exception.hpp"
namespace llframe::blas::extension::gpu {
template <is_Arithmetic X, is_Arithmetic Y>
static constexpr void multiply_vv(const int n, X *x, const int incx, Y *y,
                                  const int incy, const cudaDeviceProp &prop) {
    if (n == 0) return;
    if (incx == 1 && incy == 1) {
        if constexpr (is_Same_Ty<int8_t, X, Y>) {
            cuda::multiply_vv_i8i8(n, x, y, prop);
            return;
        }
        if constexpr (is_Same_Ty<int32_t, X, Y>) {
            cuda::multiply_vv_i32i32(n, x, y, prop);
            return;
        }
        if constexpr (is_Same_Ty<float, X, Y>) {
            cuda::multiply_vv_f32f32(n, x, y, prop);
            return;
        }
        if constexpr (is_Same_Ty<double, X, Y>) {
            cuda::multiply_vv_f64f64(n, x, y, prop);
            return;
        }
        __THROW_UNIMPLEMENTED__
    } else {
        __THROW_UNIMPLEMENTED__
    }
}

} // namespace llframe::blas::extension::gpu

#endif // LLFRAME_BLAS_BLAS_EXTENSION_GPU_MULTIPLY_VV_HPP