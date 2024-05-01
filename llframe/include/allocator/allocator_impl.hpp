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
 * @brief 分配器实现文件
 *
 */
#ifndef __LLFRAME_ALLOCATOR_IMPL_HPP__
#define __LLFRAME_ALLOCATOR_IMPL_HPP__
#include <memory>
#include <mutex>
#include <deque>
#include "core/exception.hpp"
#include "device/device_platform.hpp"
#include "allocator/allocator_define.hpp"
namespace llframe ::allocator {

// 设备分配器实现接口
template <class Ty, device::is_Device Device>
class Allocator_Impl : public Allocator_Base<Ty> {
public:
    using Self = Allocator_Impl<Ty, Device>;
    using Base = Allocator_Base<Ty>;
    using basic_allocator = Biasc_Allocator<Ty>;

    using value_type = typename Base::value_type;
    using pointer = typename Base::pointer;
    using const_pointer = typename Base::const_pointer;
    using size_type = typename Base::size_type;
    using difference_type = typename Base::difference_type;
    using void_pointer = typename Base::void_pointer;

    using device_type = Device;
    using platform = device::Device_Platform<device_type>;

public:
    [[nodiscard]] static constexpr pointer
    allocate(const size_type n, const size_type device_id = 0) {
        __THROW_UNIMPLEMENTED__;
    };

    [[nodiscard]] static constexpr void_pointer
    allocate_bytes(const size_type bytes, const size_type device_id = 0) {
        __THROW_UNIMPLEMENTED__;
    };

    static constexpr void deallocate(const pointer adress, const size_type n,
                                     const size_type device_id = 0) {
        __THROW_UNIMPLEMENTED__;
    };

    static constexpr void deallocate_bytes(const void_pointer adress,
                                           const size_type bytes,
                                           const size_type device_id = 0) {
        __THROW_UNIMPLEMENTED__;
    };
};

// CPU特化
template <class Ty>
class Allocator_Impl<Ty, device::CPU> : public Allocator_Base<Ty> {
public:
    using Self = Allocator_Impl<Ty, device::CPU>;
    using Base = Allocator_Base<Ty>;
    using basic_allocator = Biasc_Allocator<Ty>;

    using value_type = typename Base::value_type;
    using pointer = typename Base::pointer;
    using const_pointer = typename Base::const_pointer;
    using size_type = typename Base::size_type;
    using difference_type = typename Base::difference_type;
    using void_pointer = typename Base::void_pointer;

    using device_type = device::CPU;
    using platform = device::Device_Platform<device_type>;

public:
    [[nodiscard]] static constexpr pointer
    allocate(const size_type n, const size_type device_id = 0) {
        return basic_allocator::allocate(n);
    };

    [[nodiscard]] static constexpr void_pointer
    allocate_bytes(const size_type bytes, const size_type device_id = 0) {
        return basic_allocator::allocate_bytes(bytes);
    };

    static constexpr void deallocate(const pointer adress, const size_type n,
                                     const size_type device_id = 0) {
        return basic_allocator::deallocate(adress, n);
    };

    static constexpr void deallocate_bytes(const void_pointer adress,
                                           const size_type bytes,
                                           const size_type device_id = 0) {
        return basic_allocator::deallocate_bytes(adress, bytes);
    };
};

// GPU特化
template <class Ty>
class Allocator_Impl<Ty, device::GPU> : public Allocator_Base<Ty> {
public:
    using Self = Allocator_Impl<Ty, device::GPU>;
    using Base = Allocator_Base<Ty>;
    using basic_allocator = Biasc_Allocator<Ty>;

    using value_type = typename Base::value_type;
    using pointer = typename Base::pointer;
    using const_pointer = typename Base::const_pointer;
    using size_type = typename Base::size_type;
    using difference_type = typename Base::difference_type;
    using void_pointer = typename Base::void_pointer;

    using device_type = device::GPU;
    using platform = device::Device_Platform<device_type>;

protected:
    using Base::get_size;

public:
    [[nodiscard]] static constexpr pointer
    allocate(const size_type n, const size_type device_id = 0) {
        auto bytes = get_size<sizeof(value_type)>(n);
        return static_cast<pointer>(allocate_bytes(bytes, device_id));
    };

    [[nodiscard]] static constexpr void_pointer
    allocate_bytes(const size_type bytes, const size_type device_id = 0) {
        if (!platform::awake_device(device_id)) {
            __LLFRAME_THROW_EXCEPTION_INFO__(exception::Unhandled,
                                             "awake device fault!")
        }
        void *adress;
        if (cudaMalloc(&adress, bytes)) { __LLFRAME_THROW_CUDA_ERROR__; }
        return adress;
    };

    static constexpr void deallocate(const pointer adress, const size_type n,
                                     const size_type device_id = 0) {
        auto bytes = get_size<sizeof(value_type)>(n);
        return deallocate_bytes(adress, bytes, device_id);
    };

    static constexpr void deallocate_bytes(const void_pointer adress,
                                           const size_type bytes,
                                           const size_type device_id = 0) {
        if (!platform::awake_device(device_id))
            __LLFRAME_THROW_EXCEPTION_INFO__(exception::Unhandled,
                                             "awake device fault!");
        if (cudaFree(adress)) __LLFRAME_THROW_CUDA_ERROR__;
        return;
    };
};

// 分配器设置
struct Allocator_Config {
    using size_type = size_t;
    // 最小分配块大小
    static inline size_type min_block_bytes = 1024;
    // 分配块大小是2的几次幂
    static inline size_type min_block_shift = 10;
    // 偏移,辅助计算
    static inline size_type min_block_offset = 1023;

    // 重新设置分配属性
    static constexpr void reset(const size_type block_bytes) {
        block_bytes -= 1;
        min_block_bytes = 1;
        min_block_shift = 0;
        while (block_bytes) {
            block_bytes >>= 1;
            min_block_bytes <<= 1;
            min_block_shift += 1;
        }
        min_block_offset = min_block_bytes - 1;
    }

    // 调整分配大小
    static constexpr size_type adjust_bytes(const size_type bytes) {
        if (bytes & min_block_offset) {
            bytes = ((bytes >> min_block_shift) + 1) << min_block_shift;
        }
        return bytes;
    }
};

/**
 * @brief 内存分配器
 *
 *
 * @tparam Ty 类型
 * @tparam Device 设备类型
 */
template <class Ty, device::is_Device Device>
class Allocator : public Allocator_Base<Ty> {
public:
    using Self = Biasc_Allocator<Ty>;
    using Base = Allocator_Base<Ty>;

    using value_type = typename Base::value_type;
    using pointer = typename Base::pointer;
    using const_pointer = typename Base::const_pointer;
    using size_type = typename Base::size_type;
    using difference_type = typename Base::difference_type;
    using void_pointer = typename Base::void_pointer;
    using shared_pointer = std::shared_ptr<value_type>;

    using device_type = Device;
    using allocator_impl = Allocator_Impl<value_type, device_type>;
    using memory_pool = Memory_Pool<device_type>;
    using buffer_list_type = std::deque<void_pointer>;
    using config = Allocator_Config;

private:
    // 智能指针的删除器
    struct Deleter {
        constexpr Deleter(buffer_list_type &slot_ref) : _slot_ref(slot_ref){};

        // 将内存返回内存池
        constexpr void operator()(void *buffer) const {
            std::lock_guard<std::mutex> guard(al_mutex);
            _slot_ref.push_back(buffer);
        }

    private:
        buffer_list_type &_slot_ref;
    };

protected:
    using Base::get_size;

public:
    [[nodiscard]] static constexpr shared_pointer
    allocate(const size_type n, const size_type device_id = 0) {
        if (n == 0) { return shared_pointer{nullptr}; };
        auto bytes = get_size<sizeof(value_type)>(n);
        // 调整分配大小为比bytes大的最小min_block_offset整数倍
        bytes = config::adjust_bytes(bytes);
        std::lock_guard<std::mutex> guard(al_mutex);
        auto &buffer_slot = memory_pool::get_instance(device_id)[bytes];
        void_pointer buffer;
        if (buffer_slot.empty()) {
            buffer = allocator_impl::allocate_bytes(bytes, device_id);
        } else {
            buffer = buffer_slot.back();
            buffer_slot.pop_back();
        }
        return shared_pointer(static_cast<pointer>(buffer),
                              Deleter(buffer_slot));
    }

    static constexpr void deallocate_bytes(const void_pointer adress,
                                           const size_type bytes,
                                           const size_type device_id) {
        return allocator_impl::deallocate_bytes(adress, bytes, device_id);
    }

public:
    // static inline
protected:
    static inline std::mutex al_mutex;
};

} // namespace llframe::allocator

#endif //__LLFRAME_ALLOCATOR_IMPL_HPP__