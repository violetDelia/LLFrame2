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
 * @brief 张量操作器 实现
 *
 */
#ifndef LLFRAME_TENSOR_TENSOR_OPERATOR_IMPL_HPP
#define LLFRAME_TENSOR_TENSOR_OPERATOR_IMPL_HPP
#include "tensor/tensor_define.hpp"
#include "core/exception.hpp"
#include "blas/blas_define.hpp"
#include "device/device_define.hpp"

#define __LLFRAME_TRANS_VAL_ADAPTER_DEVICE(val, Pointer_T, Device_T)

namespace llframe::tensor {
/**
 * @brief 张量之间进行实际计算的实现
 *
 * @tparam N_Dim
 * @tparam Ty
 * @tparam Device
 * @version 1.0.0
 * @author 时光丶人爱 (1152488956.com)
 * @date 2024-05-18
 * @copyright Copyright (c) 2024 时光丶人爱
 */

// tensor_operator 参数检查
class _Tensor_Operator_Checker {
private:
    template <is_Tensor Tensor>
    using shape_type = typename Tensor::shape_type;
    template <is_Tensor Tensor>
    using value_type = typename Tensor::value_type;
    template <is_Tensor Tensor>
    using device_type = typename Tensor::device_type;

protected:
    template <is_Tensor... Tensors>
    static constexpr void _ensure_continious(Tensors &...tensors) {
        if constexpr (sizeof...(Tensors) == 0) return;
        if ((... || (!tensors.is_continuous()))) {
            __LLFRAME_THROW_EXCEPTION_INFO__(llframe::exception::Unimplement,
                                             "tensor is not continuous!")
        }
    }

    template <is_Tensor Tensor, is_Tensor... Others>
    static constexpr void _ensure_same_device(Tensor tensor,
                                              Others &...others) {
        if constexpr (sizeof...(Others) == 0) return;
        if constexpr (!(...
                        || is_Same_Ty<device_type<Tensor>,
                                      device_type<Others>>)) {
            __LLFRAME_THROW_EXCEPTION_INFO__(llframe::exception::Bad_Parameter,
                                             "tensor device is not same!")
        }
    };

    template <is_Tensor Tensor, is_Tensor... Others>
    static constexpr void _ensure_same_dims(Tensor tensor, Others &...others) {
        if constexpr (sizeof...(Others) == 0) return;
        if constexpr ((...
                       || (shape_type<Tensor>::Dims
                           != shape_type<Others>::Dims))) {
            __LLFRAME_THROW_EXCEPTION_INFO__(llframe::exception::Bad_Parameter,
                                             "tensor dims is not same!")
        }
    }

    template <is_Tensor Tensor, is_Tensor... Others>
    static constexpr void _ensure_same_shape(Tensor tensor, Others &...others) {
        if constexpr (sizeof...(Others) == 0) return;
        if ((... || (tensor.shape() != others.shape()))) {
            __LLFRAME_THROW_EXCEPTION_INFO__(llframe::exception::Unimplement,
                                             "tensor shape is not same!")
        }
    }
};

/**
 * @brief 不涉及主机变量的运算实现
 *
 * @version 1.0.0
 * @author 时光丶人爱 (1152488956.com)
 * @date 2024-05-18
 * @copyright Copyright (c) 2024 时光丶人爱
 * @note 变量运算的时候要唤醒设备
 */
class _Tensor_Operator_Impl_Base : public _Tensor_Operator_Checker {
private:
    using Self = _Tensor_Operator_Impl_Base;
    using Base = _Tensor_Operator_Checker;

    template <is_Tensor Tensor>
    using shape_type = typename Tensor::shape_type;
    template <is_Tensor Tensor>
    using value_type = typename Tensor::value_type;
    template <is_Tensor Tensor>
    using device_type = typename Tensor::device_type;
    template <is_Tensor Tensor>
    using blas = blas::Blas_Adapter<device_type<Tensor>>;
    template <is_Tensor Tensor>
    using platform = device::Device_Platform<device_type<Tensor>>;

protected:
    using Base::_ensure_continious;
    using Base::_ensure_same_device;
    using Base::_ensure_same_dims;
    using Base::_ensure_same_shape;

public:
    template <is_Tensor Left, is_Tensor Right>
    static constexpr void add(Left &left, Right &right) {
        __LLFRAME_TRY_CATCH_BEGIN__
        _ensure_same_device(left, right);
        _ensure_same_dims(left, right);
        _ensure_continious(left, right);
        _ensure_same_shape(left, right);
        blas<Left>::axpy(left.count(), 1, right.data(), 0, left.data(), 1,
                         platform<Left>::get_device(left.get_device_id()));
        __LLFRAME_TRY_CATCH_END__
    }

    template <is_Tensor Left, is_Tensor Right>
    static constexpr void substract(Left &left, Right &right) {
        __LLFRAME_TRY_CATCH_BEGIN__
        _ensure_same_device(left, right);
        _ensure_same_dims(left, right);
        _ensure_continious(left, right);
        _ensure_same_shape(left, right);
        blas<Left>::axpy(left.count(), -1, right.data(), 0, left.data(), 1,
                         platform<Left>::get_device(left.get_device_id()));
        __LLFRAME_TRY_CATCH_END__
    }

    template <is_Tensor Left, is_Tensor Right>
    static constexpr void divide(Left &left, Right &right) {
        __LLFRAME_TRY_CATCH_BEGIN__
        _ensure_same_device(left, right);
        _ensure_same_dims(left, right);
        _ensure_continious(left, right);
        _ensure_same_shape(left, right);
        blas<Left>::divide_vv(left.count(), right.data(), 1, left.data(), 1,
                              platform<Left>::get_device(left.get_device_id()));
        __LLFRAME_TRY_CATCH_END__
    }

    template <is_Tensor Left, is_Tensor Right>
    static constexpr void multiply(Left &left, Right &right) {
        __LLFRAME_TRY_CATCH_BEGIN__
        _ensure_same_device(left, right);
        _ensure_same_dims(left, right);
        _ensure_continious(left, right);
        _ensure_same_shape(left, right);
        blas<Left>::multiply_vv(
            left.count(), right.data(), 1, left.data(), 1,
            platform<Left>::get_device(left.get_device_id()));
        __LLFRAME_TRY_CATCH_END__
    }

    template <is_Tensor Left, is_Tensor Right>
    static constexpr void dot(Left &left, Right &right) {
        __LLFRAME_TRY_CATCH_BEGIN__
        _ensure_same_device(left, right);
        _ensure_same_dims(left, right);
        _ensure_continious(left, right);
        _ensure_same_shape(left, right);
        __THROW_UNIMPLEMENTED__;
        // blas<Left>::gemm();
        __LLFRAME_TRY_CATCH_END__
    }
};

/**
 * @brief 涉及主机变量与设备变量转换的的运算
 *
 * @version 1.0.0
 * @author 时光丶人爱 (1152488956.com)
 * @date 2024-05-18
 * @copyright Copyright (c) 2024 时光丶人爱
 * @note 需要一个辅助类
 */
class _Tensor_Operator_Impl_With_Trans_Val : public _Tensor_Operator_Checker {
private:
    using Self = _Tensor_Operator_Impl_With_Trans_Val;
    using Base = _Tensor_Operator_Checker;

    template <is_Tensor Tensor>
    using shape_type = typename Tensor::shape_type;
    template <is_Tensor Tensor>
    using value_type = typename Tensor::value_type;
    template <is_Tensor Tensor>
    using device_type = typename Tensor::device_type;
    template <is_Tensor Tensor>
    using blas = blas::Blas_Adapter<device_type<Tensor>>;

protected:
    using Base::_ensure_continious;
    using Base::_ensure_same_device;
    using Base::_ensure_same_dims;
    using Base::_ensure_same_shape;
};

class _Tensor_Operator_Impl : public _Tensor_Operator_Impl_With_Trans_Val,
                              public _Tensor_Operator_Impl_Base {};

} // namespace llframe::tensor
#endif // LLFRAME_TENSOR_TENSOR_OPERATOR_IMPL_HPP