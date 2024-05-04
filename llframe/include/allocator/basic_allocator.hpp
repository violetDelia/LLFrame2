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
 * @brief 基础分配器实现
 *
 */
#ifndef __LLFRAME_BASIC_ALLOCATOR_HPP__
#define __LLFRAME_BASIC_ALLOCATOR_HPP__
#include "allocator/allocator_define.hpp"
#include "core/base_type.hpp"
#include "core/exception.hpp"
#include <limits>
namespace llframe::allocator {

/**
 * @brief 分配器的共有属性
 */
template <class Ty>
class _Allocator_Base {
private:
    using Self = _Allocator_Base<Ty>;

public:
    using value_type = Ty;
    using pointer = Ty *;
    using const_pointer = const pointer;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using void_pointer = void *;

protected:
    /**
     * @brief 获取实际分配的字节数
     * @tparam Ty_Size 类型字节数
     * @param n 元素数量
     * @exception exception::Bad_Alloc
     * @return
     */
    template <size_type Ty_Size>
    static constexpr size_type get_size_(const size_type n) {
        if constexpr (Ty_Size == 0) return 0;
        constexpr size_type max_n =
            std::numeric_limits<size_type>::max() / Ty_Size;
        if (max_n < n)
            __LLFRAME_THROW_EXCEPTION_INFO__(exception::Bad_Alloc,
                                             "allocate betys overflow!");
        return n * Ty_Size;
    }
};

/**
 * @brief 基础分配器
 *
 *
 * @tparam Ty
 */
template <class Ty>
class Biasc_Allocator : public _Allocator_Base<Ty> {
private:
    using Self = Biasc_Allocator<Ty>;
    using Base = _Allocator_Base<Ty>;

public:
    using value_type = typename Base::value_type;
    using pointer = typename Base::pointer;
    using const_pointer = typename Base::const_pointer;
    using size_type = typename Base::size_type;
    using difference_type = typename Base::difference_type;
    using void_pointer = typename Base::void_pointer;

protected:
    using Base::get_size_;

public:
    /**
     * @brief 分配存放若干个连续元素的内存
     * @exception Bad_Alloc
     */
    [[nodiscard]] static constexpr pointer allocate(const size_type n) {
        auto bytes = get_size_<sizeof(value_type)>(n);
        return static_cast<pointer>(allocate_bytes(bytes));
    };

    /**
     * @brief 分配若干个字节的连续内存
     */
    [[nodiscard]] static constexpr void_pointer
    allocate_bytes(const size_type bytes) {
        return bytes == 0 ? nullptr : ::operator new(bytes);
    };

    /**
     * @brief 释放指定地址的内存
     * @param adress 内存地址
     * @param n 内存中元素个数
     */
    static constexpr void deallocate(const pointer adress, const size_type n) {
        ::operator delete(adress);
    };

    /**
     * @brief 释放指定地址的内存
     * @param adress 内存地址
     * @param bytes 内存中的字节数
     */
    static constexpr void deallocate_bytes(const void_pointer adress,
                                           const size_type bytes) {
        ::operator delete(adress);
    };
};

} // namespace llframe::allocator
#endif //__LLFRAME_BASIC_ALLOCATOR_HPP__