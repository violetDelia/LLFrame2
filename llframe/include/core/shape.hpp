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
 * @brief 表示形状或者尺寸的类
 *
 */
#ifndef __LLFRAME_SHAPE_HPP__
#define __LLFRAME_SHAPE_HPP__
#include "core/base_type.hpp"
#include "core/exception.hpp"
#include <array>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <iostream>
namespace llframe { inline namespace exception {
template <size_t N_Dim>

/**
 * @brief 表示尺寸或者形状的类
 *
 */
class Shape : protected std::array<int64_t, N_Dim> {
public:
    using Self = Shape<N_Dim>;
    using Base = std::array<int64_t, N_Dim>;

    using value_type = Base::value_type;
    using size_type = Base::size_type;
    using difference_type = Base::difference_type;
    using pointer = Base::pointer;
    using const_pointer = Base::const_pointer;
    using reference = Base::reference;
    using const_reference = Base::const_reference;

    using iterator = Base::iterator;
    using const_iterator = Base::const_iterator;
    using reverse_iterator = Base::reverse_iterator;
    using const_reverse_iterator = Base::const_reverse_iterator;

public: // 构造函数
    using Base::array;

    constexpr Shape(const Self &other) : Base(other) {
    }

    // 移动构造将元素置为0
    constexpr Shape(Self &&other) : Base(std::move(other)) {
        other.fill(0);
    }

    template <std::forward_iterator Iterator>
    constexpr Shape(Iterator first, Iterator last) {
        auto distance = std::distance(first, last);
        if (distance > N_Dim) {
            __LLFRAME_THROW_EXCEPTION_INFO__(
                Bad_Range, "iterator distance out of Shape range!")
        }
        if (distance < 0) {
            __LLFRAME_THROW_EXCEPTION_INFO__(Bad_Range,
                                             "iterator distance is negative!")
        }
        this->fill(0);
        std::uninitialized_copy(first, last, this->data());
    }

    template <is_Integral... Integrals>
    constexpr Shape(Integrals... values) : Base(values...){};

public: // 继承的父类方法
    using Base::at;
    using Base::size;
    using Base::data;
    using Base::fill;

    using Base::begin;
    using Base::end;
    using Base::cbegin;
    using Base::cend;
    using Base::rbegin;
    using Base::rend;
    using Base::crbegin;
    using Base::crend;

public: // 重载运算符
    [[nodiscard]] reference operator[](size_type pos) {
        return this->at(pos);
    }

    [[nodiscard]] const_reference operator[](size_type pos) const {
        return this->at(pos);
    }

    Self &operator=(const Self &other) {
        std::copy_n(other.cbegin(), N_Dim, begin());
        return *this;
    }

    Self &operator=(Self &&other) {
        this->fill(0);
        this->swap(other);
        return *this;
    }

public:
    /**
     * @brief 返回Shape所有元素相乘的结果
     */
    [[nodiscard]] value_type count() const noexcept {
        if constexpr (N_Dim == 0) { return 0; }
        return std::accumulate(this->cbegin(), this->cend(), size_type{1},
                               std::multiplies<value_type>());
    }

    /**
     * @brief 返回Shape的维度个数
     */
    [[nodiscard]] size_type dims() const noexcept {
        return Dims;
    }

public:
    static const size_type Dims = N_Dim;
};

template <class _Ty>
struct _Is_Shape : std::false_type {};

template <template <size_t> class _Ty, size_t _Arg>
struct _Is_Shape<_Ty<_Arg>> : std::is_base_of<Shape<_Arg>, _Ty<_Arg>> {};

template <class Ty>
concept is_Shape = _Is_Shape<Ty>::value;
/**
 * @brief 输出Shape的信息
 *
 */
template <is_Shape Shape>
std::ostream &operator<<(std::ostream &os, const Shape &shape) {
    auto size = shape.size();
    os << "shape: [";
    if (size > 0) {
        size_t i{};
        for (; i < size - 1; i++) { os << shape[i] << ","; }
        os << shape[i];
    }
    os.put(']');
    return os;
}

/**
 * @brief 判断两个Shape是否相等
 *
 */
template <is_Shape Left, is_Shape Right>
[[nodiscard]] constexpr bool operator==(const Left &left, const Right &right) {
    return false;
}

template <is_Shape Shape>
[[nodiscard]] constexpr bool operator==(const Shape &left, const Shape &right) {
    return std::equal(left.cbegin(), left.cend(), right.cbegin());
}

template <is_Integral... Integrals>
constexpr Shape<sizeof...(Integrals)> make_shape(Integrals... values) {
    return Shape<sizeof...(Integrals)>(values...);
}
}} // namespace llframe::exception

#endif //__LLFRAME_SHAPE_HPP__