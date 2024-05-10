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
 * @brief 内存萃取器
 *
 */
#ifndef __LLFRAME_MEMORY_FEATURES_HPP__
#include "memory/memory_define.hpp"
#include "allocator/allocator_define.hpp"
#include "blas/blas_define.hpp"
namespace llframe::memory {
/**
 * @brief  Memory的通用属性
 */
template <class Ty, device::is_Device Device>
struct Memory_Features {
    using device_type = Device;
    using size_type = size_t;

    using value_type = Ty;
    using pointer = Ty *;
    using shared_pointer = std::shared_ptr<Ty>;
    using const_pointer = const Ty *;
    using difference_type = ptrdiff_t;

    using platform = device::Device_Platform<Device>;
    using blas_adapter = blas::Blas_Adapter<Device>;
    using allocator = allocator::Allocator<Ty, Device>;
    using handle = Memory_Operator<Ty, Device>;
};

} // namespace llframe::memory
#define __LLFRAME_MEMORY_FEATURES_HPP__
#endif //__LLFRAME_MEMORY_FEATURES_HPP__