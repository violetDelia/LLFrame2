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
 * @brief 张量通用属性 文件
 *
 */
#ifndef __LLFRAME_TENSOR_FEATURES_HPP__
#define __LLFRAME_TENSOR_FEATURES_HPP__
#include "tensor/tensor_define.hpp"
#include "core/shape.hpp"
#include "blas/blas_define.hpp"
#include "memory/memory_define.hpp"
#include "device/device_define.hpp"
#include <initializer_list>
namespace llframe::tensor {

template <class _Ty, size_t _N_Dim>
struct _Tensor_Initializer_List {
    using type = std::initializer_list<
        typename _Tensor_Initializer_List<_Ty, _N_Dim - 1>::type>;
};

template <class _Ty>
struct _Tensor_Initializer_List<_Ty, 0> {
    using type = _Ty;
};

/**
 * @brief 张量的通用属性
 *
 * @tparam N_Dim 维度
 * @tparam Ty 类型
 * @tparam Device 设备类型
 */
template <size_t N_Dim, class Ty, device::is_Device Device>
struct Tensor_Features {
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    using value_type = Ty;
    using pointer = Ty *;
    using const_pointer = const Ty *;
    using reference = Ty &;
    using const_reference = const Ty &;

    using device_type = Device;
    using shape_type = shape::Shape<N_Dim>;
    using stride_type = shape::Shape<N_Dim>;
    using layout_type = blas::Blas_Layout;
    using init_list_type =
        typename _Tensor_Initializer_List<value_type, N_Dim>::type;

    using memory_type = memory::Memory<Ty, Device>;
    using platform = device::Device_Platform<Device>;
    using blas_adapter = blas::Blas_Adapter<Device>;
};
} // namespace llframe::tensor
#endif //__LLFRAME_TENSOR_FEATURES_HPP__