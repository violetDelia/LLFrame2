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
 * @brief 分配器定义文件
 *
 */
#ifndef LLFRAME_ALLOCATOR_ALLOCATOR_DEFINE_HPP
#define LLFRAME_ALLOCATOR_ALLOCATOR_DEFINE_HPP
#include "device/device_define.hpp"
#include "core/base_type.hpp"
#include <memory>
namespace llframe ::allocator {

template <class Ty>
class Biasc_AllocATor;

template <device::is_Device Device>
class Memory_Pool;

class Allocator_Config;

template <class Ty, device::is_Device Device>
class Allocator;

template <class Ty, device::is_Device Device>
struct Allocator_Features;
} // namespace llframe::allocator

#endif // LLFRAME_ALLOCATOR_ALLOCATOR_DEFINE_HPP