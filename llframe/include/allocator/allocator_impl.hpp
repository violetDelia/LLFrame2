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
#include "allocator/basic_allocator.hpp"
namespace llframe ::allocator {

template <class Ty, device::is_Device Device>
struct Allocator_Traits {
    using basic_allocator = Biasc_Allocator<Ty>;
    using value_type = Ty;
    using pointer = Ty *;
    using shared_pointer = std::shared_ptr<Ty>;
    using const_pointer = const Ty *;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using void_pointer = void *;
    using device_type = Device;
    using platform = device::Device_Platform<Device>;
    using memory_pool = Memory_Pool<Device>;
    using config = Allocator_Config;
};

/**
 * @brief 设备分配器实现接口
 *
 * @tparam Ty 元素类型
 * @tparam Device 装置类型
 */
template <class Ty, device::is_Device Device>
class _Allocator_Impl : public _Allocator_Base<Ty> {
private:
    using Self = _Allocator_Impl<Ty, Device>;
    using Base = _Allocator_Base<Ty>;
    using traits = Allocator_Traits<Ty, Device>;

public:
    using basic_allocator = typename traits::basic_allocator;
    using value_type = typename traits::value_type;
    using pointer = typename traits::pointer;
    using const_pointer = typename traits::const_pointer;
    using size_type = typename traits::size_type;
    using difference_type = typename traits::difference_type;
    using void_pointer = typename traits::void_pointer;
    using device_type = typename traits::device_type;
    using platform = typename traits::platform;

public:
    /**
     * @brief 在指定设备分配存放若干个元素连续的内存
     * @param n 元素个数
     * @param device_id 设备编号
     * @exception Unimplement
     */
    [[nodiscard]] static constexpr pointer
    allocate(const size_type n, const size_type device_id = 0) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 在指定设备分配存放若干字节的连续内存
     *
     * @param bytes – 字节数
     * @param device_id 设备编号
     * @exception Unimplement
     */
    [[nodiscard]] static constexpr void_pointer
    allocate_bytes(const size_type bytes, const size_type device_id = 0) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 释放指定设备上的内存
     *
     * @param adress 内存地址
     * @param n 元素个数
     * @param device_id 设备编号
     * @exception Unimplement
     */
    static constexpr void deallocate(const pointer adress, const size_type n,
                                     const size_type device_id = 0) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 释放指定设备上的内存
     *
     * @param adress 内存地址
     * @param bytes 字节数
     * @param device_id 设备编号
     * @exception Unimplement
     */
    static constexpr void deallocate_bytes(const void_pointer adress,
                                           const size_type bytes,
                                           const size_type device_id = 0) {
        __THROW_UNIMPLEMENTED__;
    };
};

// CPU特化
template <class Ty>
class _Allocator_Impl<Ty, device::CPU> : public _Allocator_Base<Ty> {
private:
    using Self = _Allocator_Impl<Ty, device::CPU>;
    using Base = _Allocator_Base<Ty>;
    using traits = Allocator_Traits<Ty, device::CPU>;

public:
    using basic_allocator = typename traits::basic_allocator;
    using value_type = typename traits::value_type;
    using pointer = typename traits::pointer;
    using const_pointer = typename traits::const_pointer;
    using size_type = typename traits::size_type;
    using difference_type = typename traits::difference_type;
    using void_pointer = typename traits::void_pointer;
    using device_type = typename traits::device_type;
    using platform = typename traits::platform;

public:
    /**
     * @brief 在指定设备分配存放若干个元素连续的内存
     * @param n 元素个数
     * @param device_id 设备编号
     * @exception Bad_Alloc
     */
    [[nodiscard]] static constexpr pointer
    allocate(const size_type n, const size_type device_id = 0) {
        return basic_allocator::allocate(n);
    };

    /**
     * @brief 在指定设备分配存放若干字节的连续内存
     *
     * @param bytes – 字节数
     * @param device_id 设备编号
     */
    [[nodiscard]] static constexpr void_pointer
    allocate_bytes(const size_type bytes, const size_type device_id = 0) {
        return basic_allocator::allocate_bytes(bytes);
    };

    /**
     * @brief 释放指定设备上的内存
     *
     * @param adress 内存地址
     * @param n 元素个数
     * @param device_id 设备编号
     */
    static constexpr void deallocate(const pointer adress, const size_type n,
                                     const size_type device_id = 0) {
        return basic_allocator::deallocate(adress, n);
    };

    /**
     * @brief 释放指定设备上的内存
     *
     * @param adress 内存地址
     * @param bytes 字节数
     * @param device_id 设备编号
     */
    static constexpr void deallocate_bytes(const void_pointer adress,
                                           const size_type bytes,
                                           const size_type device_id = 0) {
        return basic_allocator::deallocate_bytes(adress, bytes);
    };
};

// GPU特化
template <class Ty>
class _Allocator_Impl<Ty, device::GPU> : public _Allocator_Base<Ty> {
private:
    using Self = _Allocator_Impl<Ty, device::GPU>;
    using Base = _Allocator_Base<Ty>;
    using traits = Allocator_Traits<Ty, device::GPU>;

public:
    using basic_allocator = typename traits::basic_allocator;
    using value_type = typename traits::value_type;
    using pointer = typename traits::pointer;
    using const_pointer = typename traits::const_pointer;
    using size_type = typename traits::size_type;
    using difference_type = typename traits::difference_type;
    using void_pointer = typename traits::void_pointer;
    using device_type = typename traits::device_type;
    using platform = typename traits::platform;

protected:
    using Base::get_size_;

public:
    /**
     * @brief 在指定设备分配存放若干个元素连续的内存
     * @param n 元素个数
     * @param device_id 设备编号
     * @exception Bad_Alloc,Unhandled,CUDA_Error
     */
    [[nodiscard]] static constexpr pointer
    allocate(const size_type n, const size_type device_id = 0) {
        auto bytes = get_size_<sizeof(value_type)>(n);
        return static_cast<pointer>(allocate_bytes(bytes, device_id));
    };

    /**
     * @brief 在指定设备分配存放若干字节的连续内存
     *
     * @param bytes – 字节数
     * @param device_id 设备编号
     * @exception Unhandled,CUDA_Error
     */
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

    /**
     * @brief 释放指定设备上的内存
     *
     * @param adress 内存地址
     * @param n 元素个数
     * @param device_id 设备编号
     */
    static constexpr void deallocate(const pointer adress, const size_type n,
                                     const size_type device_id = 0) {
        auto bytes = get_size_<sizeof(value_type)>(n);
        return deallocate_bytes(adress, bytes, device_id);
    };

    /**
     * @brief 释放指定设备上的内存
     *
     * @param adress 内存地址
     * @param bytes 字节数
     * @param device_id 设备编号
     */
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
class Allocator_Config {
public:
    using size_type = size_t;

public:
    // 重新设置分配属性
    static constexpr void reset(size_type block_bytes) {
        block_bytes -= 1;
        min_block_bytes_ = 1;
        min_block_shift_ = 0;
        while (block_bytes) {
            block_bytes >>= 1;
            min_block_bytes_ <<= 1;
            min_block_shift_ += 1;
        }
        min_block_offset_ = min_block_bytes_ - 1;
    }

    // 调整分配大小
    static constexpr size_type adjust_bytes(size_type bytes) {
        if (bytes & min_block_offset_) {
            bytes = ((bytes >> min_block_shift_) + 1) << min_block_shift_;
        }
        return bytes;
    }

protected:
    // 最小分配块大小
    static inline size_type min_block_bytes_ = 1024;
    // 分配块大小是2的几次幂
    static inline size_type min_block_shift_ = 10;
    // 偏移,辅助计算
    static inline size_type min_block_offset_ = 1023;
};

/**
 * @brief 内存分配器
 *
 *
 * @tparam Ty 类型
 * @tparam Device 设备类型
 */
template <class Ty, device::is_Device Device>
class Allocator : public _Allocator_Base<Ty> {
public:
    using Self = Biasc_Allocator<Ty>;
    using Base = _Allocator_Base<Ty>;
    using traits = Allocator_Traits<Ty, Device>;
    using allocator_impl = _Allocator_Impl<Ty, Device>;

public:
    using basic_allocator = typename traits::basic_allocator;
    using value_type = typename traits::value_type;
    using pointer = typename traits::pointer;
    using shared_pointer = typename traits::shared_pointer;
    using const_pointer = typename traits::const_pointer;
    using size_type = typename traits::size_type;
    using difference_type = typename traits::difference_type;
    using void_pointer = typename traits::void_pointer;
    using device_type = typename traits::device_type;
    using platform = typename traits::platform;
    using memory_pool = typename traits::memory_pool;
    using config = typename traits::config;

    using buffer_list_type = typename memory_pool::buffer_list_type;

private:
    // 智能指针的删除器
    struct Deleter_ {
        constexpr Deleter_(buffer_list_type &slot_ref) : slot_ref_(slot_ref){};

        // 将内存返回内存池
        constexpr void operator()(void *buffer) const {
            std::lock_guard<std::mutex> guard(al_mutex_);
            slot_ref_.push_back(buffer);
        }

    private:
        buffer_list_type &slot_ref_;
    };

protected:
    using Base::get_size_;

public:
    /**
     * @brief 在指定设备分配存放若干个元素连续的内存
     * @param n 元素个数
     * @param device_id 设备编号
     * @exception Bad_Alloc,Unhandled,CUDA_Error
     */
    [[nodiscard]] static constexpr shared_pointer
    allocate(const size_type n, const size_type device_id = 0) {
        if (n == 0) { return shared_pointer{nullptr}; };
        auto bytes = get_size_<sizeof(value_type)>(n);
        // 调整分配大小为比bytes大的最小min_block_offset_整数倍
        bytes = config::adjust_bytes(bytes);
        std::lock_guard<std::mutex> guard(al_mutex_);
        auto &buffer_slot = memory_pool::get_instance(device_id)[bytes];
        void_pointer buffer;
        if (buffer_slot.empty()) {
            buffer = allocator_impl::allocate_bytes(bytes, device_id);
        } else {
            buffer = buffer_slot.back();
            buffer_slot.pop_back();
        }
        return shared_pointer(static_cast<pointer>(buffer),
                              Deleter_(buffer_slot));
    }

    /**
     * @brief 释放指定设备上的内存
     *
     * @param adress 内存地址
     * @param bytes 字节数
     * @param device_id 设备编号
     */
    static constexpr void deallocate_bytes(const void_pointer adress,
                                           const size_type bytes,
                                           const size_type device_id) {
        return allocator_impl::deallocate_bytes(adress, bytes, device_id);
    }

protected:
    static inline std::mutex al_mutex_;
};

} // namespace llframe::allocator

#endif //__LLFRAMALLOCATOR_IMPL_HPP__