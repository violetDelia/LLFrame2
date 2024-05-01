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
 * @brief 分配器 头文件
 *
 */
#ifndef __LLFRAME_MEMORY_POOL_HPP__
#define __LLFRAME_MEMORY_POOL_HPP__
#include "allocator/allocator_define.hpp"
#include "unordered_map"
#include <deque>
namespace llframe::allocator {
/**
 * @brief 内存池
 *
 */
template <device::is_Device Device>
class Memory_Pool {
public:
    using Self = Memory_Pool<Device>;
    using device_type = Device;
    using value_type = void;
    using allocator = Allocator<value_type, device_type>;
    using pointer = typename allocator::pointer;
    using size_type = typename allocator::size_type;
    using buffer_list_type = typename allocator::buffer_list_type;
    using memory_pool_type = std::unordered_map<size_type, buffer_list_type>;
    using memory_pool_map_type =
        std::unordered_map<size_type, memory_pool_type>;

protected:
    constexpr Memory_Pool() = default;
    constexpr Memory_Pool(const Self &other) = delete;

public:
    virtual ~Memory_Pool() {
        for (auto &memory_pool : memory_pool_map) {
            auto device_id = memory_pool.first;
            for (auto &buffer_slot : memory_pool.second) {
                auto buffer_size = buffer_slot.first;
                for (auto &buffer : buffer_slot.second) {
                    allocator::deallocate_bytes(static_cast<pointer>(buffer),
                                                buffer_size, device_id);
                }
                buffer_slot.second.clear();
            }
        }
    }

public:
    /**
     * @brief 获取指定设备的内存池
     * @param device_id 设备id
     */
    static constexpr auto &get_instance(size_type device_id) {
        static Self instance;
        return instance.memory_pool_map[device_id];
    }

private:
    // 内存池 memory_pool_map<device_id,memory_pool<bytes,buffer_list<void*>>>
    memory_pool_map_type memory_pool_map;
};
} // namespace llframe::allocator
#endif //__LLFRAME_MEMORY_POOL_HPP__