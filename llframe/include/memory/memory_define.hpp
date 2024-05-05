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
 * @brief 内存定义
 *
 */
#ifndef __LLFRAME_MEMORY_DEFINE_HPP__
#define __LLFRAME_MEMORY_DEFINE_HPP__
#include "device/device_define.hpp"
#include <type_traits>
namespace llframe::memory {
template <class Ty, device::is_Device Device>
struct Memory_Features;

template <class Ty, device::is_Device Device>
class Memory;

template <class Ty, device::is_Device Device>
class Memory_Operator;

template <class _Ty>
struct _Is_Memory : std::false_type {};

template <template <class, class> class _Ty, class _Arg1, class _Arg2>
struct _Is_Memory<_Ty<_Arg1, _Arg2>>
    : std::is_base_of<Memory<_Arg1, _Arg2>, _Ty<_Arg1, _Arg2>> {};

/**
 * @brief 判断类型是否为Memory
 *
 * @tparam Ty
 */
template <class Ty>
concept is_Memory = _Is_Memory<Ty>::value;

} // namespace llframe::memory
#endif //__LLFRAME_MEMORY_DEFINE_HPP__