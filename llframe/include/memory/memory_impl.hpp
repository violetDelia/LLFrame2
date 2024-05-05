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
 * @brief Memory 实现
 *
 */
#ifndef __LLFRAME_MEMORY_IMPL_HPP__
#include "memory/memory_define.hpp"
#include "core/exception.hpp"
namespace llframe::memory {
template <class Ty, device::is_Device Device>
class _Memory_Base {
private:
    using Self = _Memory_Base<Ty, Device>;
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

public: // 构造函数
    constexpr _Memory_Base() noexcept {};
    constexpr _Memory_Base(const size_type n, const size_type device_id) :
        n_elements_(n), device_id_(device_id),
        memory_(allocator::allocate(n, device_id)) {
    }
    constexpr _Memory_Base(Self &&other) noexcept :
        device_id_(std::move(other.device_id_)),
        memory_(std::move(other.memory_)),
        n_elements_(std::move(other.n_elements_)) {
    }

public:
    /**
     * @brief 获取内存所属的设备id
     */
    constexpr size_type get_id() const noexcept {
        return device_id_;
    }

    /**
     * @brief 获取内存的尺寸
     */
    constexpr size_type size() const noexcept {
        return n_elements_;
    }

protected:
    // 所属设备id
    size_type device_id_{0};
    // 内存 不对外暴露
    shared_pointer memory_{nullptr};
    // 元素个数
    size_type n_elements_{0};
};

template <class Ty, device::is_Device Device>
class Memory : public _Memory_Base<Ty, Device> {
    // 便于访问其他Memory的实例
    template <class Ty_, device::is_Device Device_>
    friend class Memory;

private:
    using Self = Memory<Ty, Device>;
    using Base = _Memory_Base<Ty, Device>;
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
    using handle = typename features::handle;

public: // 构造函数
    using Base::_Memory_Base;

public: // 堆内存操作的函数
    /**
     * @brief 获取内存指定位置的元素
     * @return constexpr value_type
     */
    constexpr value_type get(const size_type pos) const;

    /**
     * @brief 在指定位置赋值
     * @param pos 位置
     * @param val 值
     */
    constexpr void set(const size_type pos, const value_type &val);

    /**
     * @brief 在指定位置赋值
     * @param pos 位置
     * @param val 值
     */
    constexpr void set(const size_type pos, const value_type &&val);

    /**
     * @brief 将内存全部元素赋值为指定值
     * @param val 值
     */
    constexpr void fill(const value_type &val) {
        fill(0, this->n_elements_, val);
    };

    /**
     * @brief 将内存全部元素赋值为指定值
     * @param val 值
     */
    constexpr void fill(const value_type &&val) {
        fill(0, this->n_elements_, val);
    };

    /**
     * @brief 将若干个连续元素赋值为指定值
     * @param pos 位置
     * @param n 个数
     * @param val 值
     */
    constexpr void fill(const size_type pos, const size_type n,
                        const value_type &&val);

    /**
     * @brief 将若干个连续元素赋值为指定值
     * @param pos 位置
     * @param n 个数
     * @param val 值
     */
    constexpr void fill(const size_type pos, const size_type n,
                        const value_type &val);

    /**
     * @brief 对所有元素进行默认构造
     *
     */
    constexpr void construct() {
        this->construct(0, this->n_elements_);
    };

    /**
     * @brief 将若干个连续元素进行默认构造
     * @param pos 位置
     * @param n 个数
     */
    constexpr void construct(const size_type pos, const size_type n);

    /**
     * @brief 对所有元素进行析构
     *
     */
    constexpr void destroy() {
        this->destroy(0, this->n_elements_);
    };

    /**
     * @brief 将若干个连续元素进行析构
     * @param pos 位置
     * @param n 个数
     */
    constexpr void destroy(const size_type pos, const size_type n);

    /**
     * @brief 从其他的Memory的元素复制到该内存中
     * @param other 其他Memory
     */
    template <is_Memory Other_Memory>
    constexpr void copy_form(const Other_Memory &other) {
        if (other.n_elements_ != this->n_elements_) {
            __LLFRAME_THROW_EXCEPTION_INFO__(exception::Bad_Parameter,
                                             "Memory capacity is not equal!")
        }
        this->copy_form(0, this->n_elements_, other, 0);
    };

    /**
     * @brief 从其他的Memory复制若干个元素到该内存指定位置
     * @param pos 该内存起始位置
     * @param n 个数
     * @param other 其他Memory
     * @param other_pos 另一个Memory的起始位置
     */
    template <is_Memory Other_Memory>
    constexpr void copy_form(const size_type pos, const size_type n,
                             const Other_Memory &other,
                             const size_type other_pos);
};

} // namespace llframe::memory
#define __LLFRAME_MEMORY_IMPL_HPP__
#endif //__LLFRAME_MEMORY_IMPL_HPP__