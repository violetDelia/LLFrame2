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
 * @brief 线性代数库相关定义
 *
 */
#ifndef __LLFRAME_BLAS_DEFINE_HPP__
#define __LLFRAME_BLAS_DEFINE_HPP__
#include <openblas/cblas.h>
#include <cublas_v2.h>
#include <type_traits>
#include "device/device_define.hpp"

namespace llframe::blas {
enum Blas_Layout { Row_Major = 101, Col_Major = 102 };
enum Blas_Transpose {
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113,
    ConjNoTrans = 114
};
enum Blas_Uplo { Upper = 121, Lower = 122 };
enum Blas_Diag { NonUnit = 131, Unit = 132 };
enum Blas_Side { Left = 141, Right = 142 };

/**
 * @brief 是否支持调用Openblas
 *
 * @tparam Device 设备类型
 * @tparam Ty 指针参数
 * @tparam Others 指针参数
 */
template <class Device, class Ty, class... Others>
concept is_Support_Openblas = std::is_same_v<Device, device::CPU>
                              && is_Same_Floating_Point<Ty, Others...>;

/**
 * @brief 是否支持调用Cublas
 *
 * @tparam Device 设备类型
 * @tparam Ty 指针参数
 * @tparam Others 指针参数
 */
template <class Device, class Ty, class... Others>
concept is_Support_Cublas = std::is_same_v<Device, device::GPU>
                            && is_Same_Floating_Point<Ty, Others...>;

template <device::is_Device Device>
struct Blas_Adapter_Traits;

template <device::is_Device Device>
class Blas_Adapter;

} // namespace llframe::blas
#endif //__LLFRAME_BLAS_DEFINE_HPP__