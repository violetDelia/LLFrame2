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
#ifndef LLFRAME_TENSOR_TENSOR_OPERATOR_HPP
#define LLFRAME_TENSOR_TENSOR_OPERATOR_HPP
#include "tensor/tensor_define.hpp"
#include "core/exception.hpp"
#include "tensor/tensor_operator_impl.hpp"
namespace llframe::tensor {
/**
 * @brief 张量之间进行实际计算的接口
 *
 * @tparam N_Dim
 * @tparam Ty
 * @tparam Device
 * @version 1.0.0
 * @author 时光丶人爱 (1152488956.com)
 * @date 2024-05-18
 * @copyright Copyright (c) 2024 时光丶人爱
 */
class Tensor_Operator {
private:
    template <is_Tensor Tensor>
    using shape_type = typename Tensor::shape_type;
    template <is_Tensor Tensor>
    using value_type = typename Tensor::value_type;
    template <is_Tensor Tensor>
    using device_type = typename Tensor::device_type;

    using imp = _Tensor_Operator_Impl;

public:
    template <is_Tensor Left, is_Tensor_or_Arith Right>
    static constexpr void add(Left &left, Right &right) {
        imp::add(left, right);
    }

    template <is_Tensor Left, is_Tensor_or_Arith Right>
    static constexpr void add(Left &&left, Right &right) {
        imp::add(left, right);
    }

    template <is_Tensor Left, is_Tensor_or_Arith Right>
    static constexpr void add(Left &left, Right &&right) {
        imp::add(left, right);
    }

    template <is_Tensor Left, is_Tensor_or_Arith Right>
    static constexpr void add(Left &&left, Right &&right) {
        imp::add(left, right);
    }
};
} // namespace llframe::tensor
#endif // LLFRAME_TENSOR_TENSOR_OPERATOR_HPP