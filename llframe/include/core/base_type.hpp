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
 * @brief 基础类型定义
 *
 */
#ifndef __LLFRAME_BASE_TYPE_HPP__
#define __LLFRAME_BASE_TYPE_HPP__
#include <cstdint>
#include <cstddef>
#include <type_traits>
namespace llframe {
inline namespace base_type {
using int8_t = std::int8_t;
using int16_t = std::int16_t;
using int32_t = std::int32_t;
using int64_t = std::int64_t;
using uint8_t = std::uint8_t;
using uint16_t = std::uint16_t;
using uint32_t = std::uint32_t;
using uint64_t = std::uint64_t;
using float32_t = float;
using float64_t = double;
using size_t = std::size_t;
using ptrdiff_t = std::ptrdiff_t;
} // namespace base_type
// 常用概念
inline namespace concepts {

/**
 * @brief 类型是否为整数
 */
template <class Ty>
concept is_Integral = std::is_integral_v<Ty>;

/**
 * @brief 类型是否为浮点型
 */
template <class Ty>
concept is_Floating_Point = std::is_floating_point_v<Ty>;

/**
 * @brief 类型是否为数值
 */
template <class Ty>
concept is_Arithmetic = std::is_arithmetic_v<Ty>;

} // namespace concepts

} // namespace llframe
#endif //__LLFRAME_BASE_TYPE_HPP__