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
 * @brief 张量定义 文件
 *
 */
#ifndef __LLFRAME_TENSOR_DEFINE_HPP__
#define __LLFRAME_TENSOR_DEFINE_HPP__
#include "core/base_type.hpp"
#include "device/device_define.hpp"
#include <type_traits>
namespace llframe::tensor {
template <size_t N_Dim, class Ty, device::is_Device Device>
struct Tensor_Features;

template <size_t N_Dim, class Ty, device::is_Device Device>
class Tensor;

template <class _Ty>
struct _Is_Tensor : std::false_type{};

template <template <size_t, class, class> class _Ty, size_t _Arg1, class _Arg2,
          class _Arg3>
struct _Is_Tensor<_Ty<_Arg1, _Arg2, _Arg3>>:std::is_base_of<
    Tensor<_Arg1, _Arg2, _Arg3>, _Ty<_Arg1, _Arg2, _Arg3>> {};

template <class Ty>
concept is_Tensor = _Is_Tensor<Ty>::value;

template <class Ty>
concept is_Tensor_or_Arith = is_Tensor<Ty> || is_Arithmetic<Ty>;

} // namespace llframe::tensor
#endif //__LLFRAME_TENSOR_DEFINE_HPP__