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
 * @brief device 实现
 *
 */
#ifndef __LLFRAME_DEVICE_IMPL_HPP__
#define __LLFRAME_DEVICE_IMPL_HPP__
#include <type_traits>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include "core/base_type.hpp"
#include "core/exception.hpp"
namespace llframe { namespace device {
/**
 * @brief 设备的基类
 *
 */
class Device {
public:
    using Self = Device;
    using size_type = size_t;

public: // 构造函数
    constexpr Device() : Device(0){};
    constexpr Device(const size_type device_id) : _id(device_id){};
    constexpr Device(const Self &other) = default;
    constexpr Device(Self &&other) = delete;
    constexpr Self &operator=(const Self &other) = default;
    constexpr Self &operator=(Self &&other) = delete;
    virtual ~Device(){};

public:
    /**
     * @brief 返回设备编号
     */
    constexpr size_type get_id() {
        return this->_id;
    }

    /**
     * @brief 唤醒该设备
     */
    virtual bool awake() = 0;

protected:
    // 设备编号
    size_type _id{};
};

/**
 * @brief CPU
 *
 */
class CPU : public Device {
public:
    using Base = Device;

public:
    using Base::Device;

public:
    bool awake() override {
        return true;
    }
};

/**
 * @brief GPU
 *
 */
class GPU : public Device {
public:
    using Self = GPU;
    using Base = Device;
    using size_type = typename Base::size_type;
    using property_type = cudaDeviceProp;
    using property_pointer = std::shared_ptr<property_type>;
    using cublas_handle_type = cublasHandle_t;
    using cublas_handle_pointer = std::shared_ptr<cublas_handle_type>;
    using cudnn_handle_type = cudnnHandle_t;
    using cudnn_handle_pointer = std::shared_ptr<cudnn_handle_type>;

private:
    using Base::Device;

public: // 重写构造函数
    GPU(const size_type device_id) : Base(device_id) {
        auto id = static_cast<int>(device_id);
        int device_count;
        if (cudaGetDeviceCount(&device_count)) __LLFRAME_THROW_CUDA_ERROR__
        if (id >= device_count)
            __LLFRAME_THROW_UNHANDLED_INFO__("device is not exist!")
        _property.reset(new property_type);
        if (cudaGetDeviceProperties(&(*_property), id))
            __LLFRAME_THROW_CUDA_ERROR__
        _default_cudnn_handle.reset(new cudnn_handle_type);
        if (cudnnCreate(&(*_default_cudnn_handle)))
            __LLFRAME_THROW_CUDNN_ERROR__
        _default_cublas_handle.reset(new cublas_handle_type);
        if (cublasCreate(&(*_default_cublas_handle)))
            __LLFRAME_THROW_CUBLAS_ERROR__
    }
    GPU() : GPU(0){};
    virtual ~GPU() {
        if (_default_cudnn_handle.use_count() == 1) {
            cudnnDestroy(*_default_cudnn_handle);
        }
        if (_default_cublas_handle.use_count() == 1) {
            cublasDestroy(*_default_cublas_handle);
        }
    }

public:
    bool awake() override {
        if (cudaSetDevice(this->_id)) { return false; }
        return true;
    };

    auto &cudnn_handle() {
        return *_default_cudnn_handle;
    }

    auto &cublas_handle() {
        return *_default_cublas_handle;
    }

    auto &property() {
        return *_property;
    }

protected:
    // 该设备的属性信息
    property_pointer _property{};
    // 调用cudnn默认的cudnnHandle_t
    cudnn_handle_pointer _default_cudnn_handle{};
    // 调用cublas默认的cublasHandle_t
    cublas_handle_pointer _default_cublas_handle{};
};

}}     // namespace llframe::device
#endif //__LLFRAME_DEVICE_IMPL_HPP__