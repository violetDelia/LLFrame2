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
 * @brief 张量基础功能实现
 *
 */
#ifndef LLFRAME_TENSOR_TENSOR_BASE_HPP
#define LLFRAME_TENSOR_TENSOR_BASE_HPP
#include "tensor/tensor_define.hpp"
#include "core/exception.hpp"
namespace llframe::tensor {

/**
 * @brief 张量基类;getter,setter和一些基础构造函数的实现
 *
 */
template <size_t N_Dim, class Ty, device::is_Device Device>
class _Tensor_Base {
private:
    using Self = _Tensor_Base<N_Dim, Ty, Device>;
    using features = Tensor_Features<N_Dim, Ty, Device>;

public:
    using size_type = typename features::size_type;
    using difference_type = typename features::difference_type;

    using value_type = typename features::value_type;
    using pointer = typename features::pointer;
    using const_pointer = typename features::const_pointer;
    using reference = typename features::reference;
    using const_reference = typename features::const_reference;

    using device_type = typename features::device_type;
    using shape_type = typename features::shape_type;
    using stride_type = typename features::stride_type;
    using layout_type = typename features::layout_type;
    using init_list_type = typename features::init_list_type;

    using memory_type = typename features::memory_type;
    using platform = typename features::platform;
    using blas_adapter = typename features::blas_adapter;

public: // 构造函数
    constexpr _Tensor_Base() :
        shape_{}, stride_{}, start_{0}, memory_{}, device_id_{0} {
    }

    explicit constexpr _Tensor_Base(const shape_type &shape,
                                    const size_type device_id = 0) :
        shape_(shape), start_{0}, memory_(this->shape_.count(), device_id),
        device_id_{device_id} {
        this->_init_stride(this->shape_);
    }

    explicit constexpr _Tensor_Base(shape_type &&shape,
                                    const size_type device_id = 0) :
        shape_(std::move(shape)), start_{0},
        memory_(this->shape_.count(), device_id), device_id_{device_id} {
        this->_init_stride(this->shape_);
    }

    constexpr _Tensor_Base(const value_type &val, const shape_type &shape,
                           const size_type device_id = 0) :
        _Tensor_Base(shape, device_id) {
        this->memory_.fill(val);
    }

    constexpr _Tensor_Base(value_type &&val, const shape_type &shape,
                           const size_type device_id = 0) :
        _Tensor_Base(shape, device_id) {
        this->memory_.fill(std::move(val));
    }

    constexpr _Tensor_Base(const value_type &val, shape_type &&shape,
                           const size_type device_id = 0) :
        _Tensor_Base(std::move(shape), device_id) {
        this->memory_.fill(val);
    }

    constexpr _Tensor_Base(value_type &&val, shape_type &&shape,
                           const size_type device_id = 0) :
        _Tensor_Base(std::move(shape), device_id) {
        this->memory_.fill(std::move(val));
    }

    constexpr _Tensor_Base(const Self &other) :
        shape_(other.shape_), stride_(other.stride_), start_(other.start_),
        layout_(other.layout_), device_id_(other.device_id_),
        memory_(other.memory_) {
    }

    constexpr _Tensor_Base(Self &&other) :
        shape_(std::move(other.shape_)), stride_(std::move(other.stride_)),
        device_id_(std::move(other.device_id_)),
        memory_(std::move(other.memory_)), layout_(std::move(other.layout_)),
        start_(std::move(other.start_)) {
        other.start_ = 0;
        other.device_id_ = 0;
    }

    virtual ~_Tensor_Base() {};

protected:
    // 根据shape初始化stride
    constexpr void _init_stride(const shape_type &shape) {
        if (shape.count() == 0) {
            stride_.fill(0);
            return;
        }
        auto stride_it = stride_.rbegin();
        const auto stride_end = stride_.rend();
        auto shape_it = shape.crbegin();
        size_type stride = 1;
        *stride_it = stride;
        stride_it++;
        while (stride_it != stride_end) {
            stride *= *shape_it;
            *stride_it = stride;
            stride_it++;
            shape_it++;
        }
    }

public:
    /**
     * @brief 获取Tensor的形状
     */
    constexpr shape_type shape() const {
        return shape_;
    }

    /**
     * @brief 获取Tensor的stride。
     */
    constexpr stride_type stride() const {
        return stride_;
    }

    /**
     * @brief 获取Tensor的真实数据内存
     * @param copy 如果为真，返回的是内存的拷贝，否则是引用。
     */
    constexpr memory_type memory(const bool copy = false) const {
        if (copy) return memory_;
        return std::move(memory_.ref());
    }

    /**
     * @brief 获取Tensor的设备编号
     */
    constexpr size_type get_device_id() const noexcept {
        return device_id_;
    }

    /**
     * @brief Tensor中的元素总数
     */
    constexpr size_type count() const {
        return shape_.count();
    }

    /**
     * @brief 检查Tensor是否连续
     * @return
     */
    constexpr bool is_continuous() const {
        size_type stride = 1;
        for (size_type i = N_Dim - 1; i > 0; --i) {
            if (stride != this->stride_[i]) return false;
            stride = stride * this->shape_[i];
        }
        return true;
    }

    /**
     * @brief 获取实际内存地址
     *
     * @return constexpr pointer
     * @version 1.0.0
     * @author 时光丶人爱 (1152488956.com)
     * @date 2024-05-18
     * @copyright Copyright (c) 2024 时光丶人爱
     */
    constexpr pointer data() const {
        return memory_.data();
    }

protected:
    // 形状
    shape_type shape_;
    stride_type stride_;
    // 数据在内存中的起始位置
    size_type start_;
    // 数据内存
    memory_type memory_;
    // 设备编号
    size_type device_id_;
    // 暂时只实现行排布的
    const layout_type layout_ = layout_type::Row_Major;
};

/**
 * @brief Tensor 初始化列表的实现
 *
 * @tparam N_Dim
 * @tparam Ty
 * @tparam Device
 */
template <size_t N_Dim, class Ty, device::is_Device Device>
class _Tensor_Init_List : public _Tensor_Base<N_Dim, Ty, Device> {
private:
    using Self = _Tensor_Init_List<N_Dim, Ty, Device>;
    using Base = _Tensor_Base<N_Dim, Ty, Device>;
    using features = Tensor_Features<N_Dim, Ty, Device>;

public:
    using size_type = typename features::size_type;
    using difference_type = typename features::difference_type;

    using value_type = typename features::value_type;
    using pointer = typename features::pointer;
    using const_pointer = typename features::const_pointer;
    using reference = typename features::reference;
    using const_reference = typename features::const_reference;

    using device_type = typename features::device_type;
    using shape_type = shape::Shape<N_Dim>;
    // using shape_type = typename features::shape_type;
    using stride_type = shape::Shape<N_Dim>;
    // using stride_type = typename features::stride_type;
    using layout_type = blas::Blas_Layout;
    // using layout_type = typename features::layout_type;
    using init_list_type = typename features::init_list_type;

    using memory_type = memory::Memory<Ty, Device>;
    // using memory_type = typename features::memory_type;
    using platform = typename features::platform;
    using blas_adapter = typename features::blas_adapter;

public:
    using Base::_Tensor_Base;
    using Base::shape;
    using Base::stride;
    using Base::memory;
    using Base::get_device_id;
    using Base::count;
    using Base::is_continuous;
    using Base::data;

private:
    template <size_type N>
    constexpr void _init_memory(
        size_type pos,
        typename Tensor_Features<N, value_type, device_type>::init_list_type
            init_list) {
        if (init_list.size() > this->shape_[N_Dim - N]) {
            __LLFRAME_THROW_EXCEPTION_INFO__(exception::Bad_Range,
                                             "init_list out of range of shape!")
        }
        auto init_list_it = init_list.begin();
        const auto init_list_end = init_list.end();
        while (init_list_it != init_list_end) {
            this->_init_memory<N - 1>(pos, *init_list_it);
            pos += this->stride_[N_Dim - N];
            init_list_it++;
        }
    };

    template <>
    constexpr void _init_memory<1>(
        const size_type pos,
        typename Tensor_Features<1, value_type, device_type>::init_list_type
            init_list) {
        if (init_list.size() > this->shape_[N_Dim - 1]) {
            __LLFRAME_THROW_EXCEPTION_INFO__(exception::Bad_Range,
                                             "init_list out of range of shape!")
        }
        this->memory_.fill(pos, init_list);
    }

public:
    constexpr _Tensor_Init_List(init_list_type init_list,
                                const shape_type &shape,
                                const size_type device_id = 0) :
        Self(shape, device_id) {
        this->_init_memory<N_Dim>(this->start_, init_list);
    };

    constexpr _Tensor_Init_List(init_list_type init_list, shape_type &&shape,
                                const size_type device_id = 0) :
        Self(std::move(shape), device_id) {
        this->_init_memory<N_Dim>(this->start_, init_list);
    };
};

/**
 * @brief Tensor 初始化列表的实现
 * 如果是0维度的会和其他初始化冲突，所以0维度特化没有使用初始化列表初始化
 *
 * @tparam N_Dim
 * @tparam Ty
 * @tparam Device
 */
template <class Ty, device::is_Device Device>
class _Tensor_Init_List<0, Ty, Device> : public _Tensor_Base<0, Ty, Device> {
private:
    using Self = _Tensor_Init_List<0, Ty, Device>;
    using Base = _Tensor_Base<0, Ty, Device>;
    using features = Tensor_Features<0, Ty, Device>;

public:
    using size_type = typename features::size_type;
    using difference_type = typename features::difference_type;

    using value_type = typename features::value_type;
    using pointer = typename features::pointer;
    using const_pointer = typename features::const_pointer;
    using reference = typename features::reference;
    using const_reference = typename features::const_reference;

    using device_type = typename features::device_type;
    using shape_type = typename features::shape_type;
    using stride_type = typename features::stride_type;
    using layout_type = typename features::layout_type;
    using init_list_type = typename features::init_list_type;

    using memory_type = typename features::memory_type;
    using platform = typename features::platform;
    using blas_adapter = typename features::blas_adapter;

public:
    using Base::_Tensor_Base;
    using Base::shape;
    using Base::stride;
    using Base::memory;
    using Base::get_device_id;
    using Base::count;
};

} // namespace llframe::tensor
#endif // LLFRAME_TENSOR_TENSOR_BASE_HPP