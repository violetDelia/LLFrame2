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
#ifndef LLFRAME_MEMORY_MEMORY_IMPL_HPP
#define LLFRAME_MEMORY_MEMORY_IMPL_HPP

#include "memory/memory_define.hpp"
#include "memory/memory_operator.hpp"
#include "core/exception.hpp"
#include <initializer_list>
#include <type_traits>
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
    constexpr _Memory_Base() {};
    constexpr _Memory_Base(const size_type n, const size_type device_id) :
        n_elements_(n), device_id_(device_id),
        memory_(allocator::allocate(n, device_id)) {
    }
    constexpr _Memory_Base(Self &&other) :
        device_id_(std::move(other.device_id_)),
        memory_(std::move(other.memory_)),
        n_elements_(std::move(other.n_elements_)) {
        other.n_elements_ = 0;
        other.device_id_ = 0;
    }

    virtual ~_Memory_Base() {};

protected:
    constexpr _Memory_Base(const size_type n_elements,
                           const shared_pointer memory,
                           const size_type device_id) :
        n_elements_(n_elements), memory_(memory), device_id_(device_id) {};

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

    /**
     * @brief 内存地址引用个数
     *
     */
    constexpr size_type use_count() const {
        return memory_.use_count();
    }

    /**
     * @brief 获取实际内存地址
     * @return
     */
    constexpr pointer data() const {
        return memory_.get();
    }

protected:
    // 所属设备id
    size_type device_id_{0};
    // 内存 不对外暴露
    shared_pointer memory_{nullptr};
    // 元素个数
    size_type n_elements_{0};
};

/**
 * @brief 表示设备上的一段连续内存
 *
 *
 * @tparam Ty 类型
 * @tparam Device 装置
 */
template <class Ty, device::is_Device Device>
class Memory : public _Memory_Base<Ty, Device> {
    // 便于访问其他Memory的实例
    template <class Ty_, device::is_Device Device_>
    friend class Memory;
    // 便于对Memory进行操作
    template <class Ty_, device::is_Device Device_>
    friend class Memory_Operator;

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
    using shared_pointer = std::shared_ptr<Ty>;
    // using shared_pointer = typename features::shared_pointer;
    using difference_type = typename features::difference_type;

    using platform = typename features::platform;
    using blas_adapter = typename features::blas_adapter;
    using allocator = typename features::allocator;
    using handle = typename features::handle;

public: // 构造函数
    constexpr Memory() : Base() {};

    constexpr Memory(const size_type n, const size_type device_id) :
        Base(n, device_id) {
        this->construct();
    }

    constexpr Memory(const Self &other) :
        Base(other.n_elements_, other.device_id_) {
        if (!is_Arithmetic<Ty>) this->construct();
        this->copy_from(other);
    }

    constexpr Memory(Self &&other) : Base(std::move(other)) {
    }

    ~Memory() override {
        if constexpr (is_Arithmetic<value_type>) return;
        if (this->n_elements_ == 0) return;
        if (this->memory_.use_count() == 1) { this->destroy(); }
    }

protected:
    using Base::_Memory_Base;

public:
    /**
     * @brief 获取内存的一个浅拷贝
     *
     * @return constexpr Self
     */
    constexpr Self ref() const {
        return {this->n_elements_, this->memory_, this->device_id_};
    }

public: // 内存操作的函数
    /**
     * @brief 获取内存指定位置的元素
     */
    constexpr value_type get(const size_type pos) const {
        return handle::get(*this, pos);
    };

    /**
     * @brief 在指定位置赋值
     * @param pos 位置
     * @param val 值
     */
    constexpr void set(const size_type pos, const value_type &val) {
        handle::set(*this, pos, val);
    };

    /**
     * @brief 在指定位置赋值
     * @param pos 位置
     * @param val 值
     */
    constexpr void set(const size_type pos, const value_type &&val) {
        handle::set(*this, pos, val);
    };

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
                        const value_type &val) {
        handle::fill(*this, pos, n, val);
    };

    /**
     * @brief 将若干个连续元素赋值为指定值
     * @param pos 位置
     * @param n 个数
     * @param val 值
     */
    constexpr void fill(const size_type pos, const size_type n,
                        value_type &&val) {
        handle::fill(*this, pos, n, val);
    };

    /**
     * @brief 对内存指定位置用初始化列表赋值
     * @param pos 位置
     * @param init_list 初始化列表
     */
    constexpr void fill(const size_type pos,
                        std::initializer_list<value_type> init_list) {
        handle::fill(*this, pos, init_list.begin(), init_list.size());
    };

    /**
     * @brief 从其他的Memory的元素复制到该内存中
     * @param other 其他Memory
     */
    template <is_Memory Other_Memory>
    constexpr void copy_from(const Other_Memory &other) {
        if (other.n_elements_ != this->n_elements_) {
            __LLFRAME_THROW_EXCEPTION_INFO__(exception::Bad_Parameter,
                                             "Memory capacity is not equal!")
        }
        this->copy_from(0, this->n_elements_, other, 0);
    };

    /**
     * @brief 从其他的Memory复制若干个元素到该内存指定位置
     * @param pos 该内存起始位置
     * @param n 个数
     * @param other 其他Memory
     * @param other_pos 另一个Memory的起始位置
     */
    template <is_Memory Other_Memory>
    constexpr void copy_from(const size_type pos, const size_type n,
                             const Other_Memory &other,
                             const size_type other_pos) {
        handle::copy_from(*this, pos, n, other, other_pos);
    };

protected:
    /**
     * @brief 对所有元素进行默认构造初始化
     *
     */
    constexpr void construct() {
        this->construct(0, this->n_elements_);
    };

    /**
     * @brief 将若干个连续元素进行默认构造初始化
     * @param pos 位置
     * @param n 个数
     */
    constexpr void construct(const size_type pos, const size_type n) {
        handle::construct(*this, pos, n);
    };

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
    constexpr void destroy(const size_type pos, const size_type n) {
        handle::destroy(*this, pos, n);
    };
};

} // namespace llframe::memory
#endif // LLFRAME_MEMORY_MEMORY_IMPL_HPP