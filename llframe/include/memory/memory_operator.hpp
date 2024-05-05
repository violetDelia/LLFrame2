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
 * @brief 对Memory进行操作的类
 *
 */
#ifndef __LLFRAME_MEMORY_OPERATOR_HPP__
#define __LLFRAME_MEMORY_OPERATOR_HPP__
#include "core/exception.hpp"
#include "memory/memory_define.hpp"
#include "blas/blas_define.hpp"
#include <algorithm>
namespace llframe::memory {
/**
 * @brief 对Memory操作进行检查,独立出来的类型,内部方法外部不可见
 */
template <class Ty, device::is_Device Device>
class Memory_Checker {
private:
    using Self = Memory_Checker<Ty, Device>;
    using features = Memory_Features<Ty, Device>;

public:
    using device_type = typename features::device_type;
    using size_type = typename features::size_type;

    using value_type = typename features::value_type;
    using pointer = typename features::pointer;
    using const_pointer = typename features::const_pointer;
    using shared_pointer = typename features::shared_pointer;
    using difference_type = typename features::difference_type;

    using platform = typename features::platform;
    using blas_adapter = typename features::blas_adapter;
    using allocator = typename features::allocator;

protected:
    /**
     * @brief 判断参数是否超出memory的范围
     *
     */
    template <is_Memory Memory>
    static constexpr void ensure_pos_legally_(const Memory &memory,
                                              const size_type pos,
                                              const size_type n) {
        if (memory.size() >= pos + n) return;
        __LLFRAME_THROW_EXCEPTION_INFO__(exception::Bad_Range,
                                         "out of Memory range!")
    }
};

/**
 * @brief 对Memory内存实际进行操作的类,方便特化和加功能
 */
template <class Ty, device::is_Device Device>
class Memory_Operator : public Memory_Checker<Ty, Device> {
private:
    using Base = Memory_Checker<Ty, Device>;
    using Self = Memory_Operator<Ty, Device>;
    using features = Memory_Features<Ty, Device>;

public:
    using device_type = typename features::device_type;
    using size_type = typename features::size_type;

    using value_type = typename features::value_type;
    using pointer = typename features::pointer;
    using const_pointer = typename features::const_pointer;
    using shared_pointer = typename features::shared_pointer;
    using difference_type = typename features::difference_type;

    using platform = typename features::platform;
    using blas_adapter = typename features::blas_adapter;
    using allocator = typename features::allocator;
    using memory_type = Memory<Ty, Device>;

protected:
    using Base::ensure_pos_legally_;

public:
    /**
     * @brief 获取内存指定位置的元素
     */
    static constexpr value_type get(const memory_type &memory,
                                    const size_type pos) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 在内存指定位置赋值
     * @param memory 内存
     * @param pos 位置
     * @param val 值
     */
    static constexpr void set(memory_type &memory, const size_type pos,
                              const value_type &val) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 将内存中若干个连续元素赋值为指定值
     * @param memory 内存
     * @param pos 位置
     * @param n 个数
     * @param val 值
     */
    static constexpr void fill(memory_type &memory, const size_type pos,
                               const size_type n, const value_type &val) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 对内存指定位置用初始化列表赋值
     * @param memory 内存
     * @param pos 位置
     * @param init_list 初始化列表
     */
    static constexpr void fill(memory_type &memory, const size_type pos,
                               std::initializer_list<value_type> init_list) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 对内存若干个连续元素进行默认初始化
     *
     * @param memory 内存
     * @param pos 位置
     * @param n 个数
     */
    static constexpr void construct(memory_type &memory, const size_type pos,
                                    const size_type n) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 对内存若干个连续元素进行析构
     *
     * @param memory 内存
     * @param pos 位置
     * @param n 个数
     */
    static constexpr void destroy(memory_type &memory, const size_type pos,
                                  const size_type n) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 从其他的Memory复制若干个元素到该内存指定位置.
     * @param to 被赋值的Memory
     * @param to_pos 被赋值Memory起始位置
     * @param n 个数
     * @param other 被复制的Memory
     * @param from_pos 被复制的Memory起始位置
     *
     * @note 为了方便不同类型不同设备赋值的接口,具体实现另写
     */
    template <is_Memory From_Memory>
    static constexpr void copy_form(memory_type &to, const size_type to_pos,
                                    const size_type n, const From_Memory &from,
                                    const size_type from_pos) {
        __THROW_UNIMPLEMENTED__;
    };
};

/**
 * @brief 对Memory内存实际进行操作的类,方便特化和加功能
 * @note CPU特化
 */
template <class Ty>
class Memory_Operator<Ty, device::CPU>
    : public Memory_Checker<Ty, device::CPU> {
private:
    using Base = Memory_Checker<Ty, device::CPU>;
    using Self = Memory_Operator<Ty, device::CPU>;
    using features = Memory_Features<Ty, device::CPU>;

public:
    using device_type = typename features::device_type;
    using size_type = typename features::size_type;

    using value_type = typename features::value_type;
    using pointer = typename features::pointer;
    using const_pointer = typename features::const_pointer;
    using shared_pointer = typename features::shared_pointer;
    using difference_type = typename features::difference_type;

    using platform = typename features::platform;
    using blas_adapter = blas::Blas_Adapter<device::CPU>;
    // using blas_adapter = typename features::blas_adapter;
    using allocator = typename features::allocator;
    using memory_type = Memory<Ty, device::CPU>;

protected:
    using Base::ensure_pos_legally_;

public:
    /**
     * @brief 获取内存指定位置的元素
     */
    static constexpr value_type get(const memory_type &memory,
                                    const size_type pos) {
        ensure_pos_legally_(memory, pos, 1);
        return *(memory.memory_.get() + pos);
    };

    /**
     * @brief 在内存指定位置赋值
     * @param memory 内存
     * @param pos 位置
     * @param val 值
     */
    static constexpr void set(memory_type &memory, const size_type pos,
                              const value_type &val) {
        ensure_pos_legally_(memory, pos, 1);
        *(memory.memory_.get() + pos) = val;
    };

    /**
     * @brief 将内存中若干个连续元素赋值为指定值
     * @param memory 内存
     * @param pos 位置
     * @param n 个数
     * @param val 值
     */
    static constexpr void fill(memory_type &memory, const size_type pos,
                               const size_type n, const value_type &val) {
        ensure_pos_legally_(memory, pos, n);
        // 如果可以,尽量调用blas
        if constexpr (blas::is_Support_Blas<device_type, value_type>) {
            blas_adapter::copy(n, &val, 0, memory.memory_.get() + pos, 1);
        } else {
            std::uninitialized_fill_n(memory.memory_.get() + pos, n, val);
        }
    };

    /**
     * @brief 对内存指定位置用初始化列表赋值
     * @param memory 内存
     * @param pos 位置
     * @param init_list 初始化列表
     */
    static constexpr void fill(memory_type &memory, const size_type pos,
                               std::initializer_list<value_type> init_list) {
        ensure_pos_legally_(memory, pos, init_list.size());
        std::uninitialized_move_n(init_list.cbegin(), init_list.size(),
                                  memory.memory_.get() + pos);
    };

    /**
     * @brief 对内存若干个连续元素进行默认初始化
     *
     * @param memory 内存
     * @param pos 位置
     * @param n 个数
     */
    static constexpr void construct(memory_type &memory, const size_type pos,
                                    const size_type n) {
        ensure_pos_legally_(memory, pos, n);
        std::uninitialized_value_construct_n(memory.memory_.get() + pos, n);
    };

    /**
     * @brief 对内存若干个连续元素进行析构
     *
     * @param memory 内存
     * @param pos 位置
     * @param n 个数
     */
    static constexpr void destroy(memory_type &memory, const size_type pos,
                                  const size_type n) {
        ensure_pos_legally_(memory, pos, n);
        std::destroy_n(memory.memory_.get() + pos, n);
    };

    /**
     * @brief 从其他的Memory复制若干个元素到该内存指定位置.
     * @param to 被赋值的Memory
     * @param to_pos 被赋值Memory起始位置
     * @param n 个数
     * @param other 被复制的Memory
     * @param from_pos 被复制的Memory起始位置
     *
     * @note 为了方便不同类型不同设备赋值的接口,具体实现另写
     */
    template <is_Memory From_Memory>
    static constexpr void copy_form(memory_type &to, const size_type to_pos,
                                    const size_type n, const From_Memory &from,
                                    const size_type from_pos) {
        __THROW_UNIMPLEMENTED__;
    };

    /**
     * @brief 从其他的Memory复制若干个元素到该内存指定位置;CPU与CPU的实现.
     * @param to 被赋值的Memory
     * @param to_pos 被赋值Memory起始位置
     * @param n 个数
     * @param other 被复制的Memory
     * @param from_pos 被复制的Memory起始位置
     *
     */
    template <class Ty>
    static constexpr void
    copy_form(memory_type &to, const size_type to_pos, const size_type n,
              const Memory<Ty, device::CPU> &from, const size_type from_pos) {
        ensure_pos_legally_(to, to_pos, n);
        ensure_pos_legally_(from, from_pos, n);
        // 如果可以,尽量调用blas
        if constexpr (blas::is_Support_Blas<device_type, value_type, Ty>) {
            blas_adapter::copy(n, from.memory_.get() + from_pos, 1,
                               to.memory_.get() + to_pos, 1);
        } else {
            std ::uninitialized_copy_n(from.memory_.get() + from_pos, n,
                                       to.memory_.get() + to_pos);
        }
    };
};

} // namespace llframe::memory

#endif //__LLFRAME_MEMORY_OPERATOR_HPP__