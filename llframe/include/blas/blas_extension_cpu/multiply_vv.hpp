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
 * @brief cpu 算子扩展  vec*vec
 *
 */

#ifndef LLFRAME_BLAS_BLAS_EXTENSION_CPU_MULTIPLY_VV_HPP
#define LLFRAME_BLAS_BLAS_EXTENSION_CPU_MULTIPLY_VV_HPP
#include "core/base_type.hpp"
namespace llframe::blas::extension::cpu {
template <is_Arithmetic X, is_Arithmetic Y>
static constexpr void multiply_vv(const int n, X *x, const int incx, Y *y,
                                  const int incy) {
    if (n == 0) return;
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; ++i) { y[i] = x[i] * y[i]; }
    } else {
        for (int i = 0; i < n; ++i) { y[i * incy] = x[i * incx] * y[i * incy]; }
    }
}

} // namespace llframe::blas::extension::cpu

#endif // LLFRAME_BLAS_BLAS_EXTENSION_CPU_MULTIPLY_VV_HPP