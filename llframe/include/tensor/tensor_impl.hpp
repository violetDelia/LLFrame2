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
 * @brief 张量实现文件
 *
 */
#ifndef __LLFRAME_TENSOR_IMPL_HPP__
#define __LLFRAME_TENSOR_IMPL_HPP__
#include "tensor/tensor_define.hpp"
#include "tensor/tensor_base.hpp"
namespace llframe::tensor {
/**
 * @brief 张量
 * @tparam Ty 类型
 * @tparam N_Dim 维度
 * @tparam Device 设备类型
 */
template <size_t N_Dim, class Ty = float32_t,
          device::is_Device Device = device::CPU>
class Tensor : public _Tensor_Init_List<N_Dim, Ty, Device> {
private:
    using Self = Tensor<N_Dim, Ty, Device>;
    using Base = _Tensor_Init_List<N_Dim, Ty, Device>;
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

public:
    using Base::_Tensor_Init_List;
    using Base::shape;
    using Base::stride;
    using Base::memory;
    using Base::get_device_id;
    using Base::count;
};

template <class Ty, device::is_Device Device>
class Tensor<0, Ty, Device> : public _Tensor_Base<0, Ty, Device> {
private:
    using Self = Tensor<0, Ty, Device>;
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
};

} // namespace llframe::tensor
#endif //__LLFRAME_TENSOR_IMPL_HPP__