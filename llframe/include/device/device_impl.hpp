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
namespace llframe ::device {
/**
 * @brief 设备的基类
 *
 */
class _Device {
public:
    using Self = _Device;
    using size_type = size_t;

public: // 构造函数
    constexpr _Device() : _Device(0){};
    explicit constexpr _Device(const size_type device_id) : id_(device_id){};
    constexpr _Device(const Self &other) = default;
    constexpr _Device(Self &&other) = delete;
    constexpr Self &operator=(const Self &other) = default;
    constexpr Self &operator=(Self &&other) = delete;
    virtual ~_Device(){};

public:
    /**
     * @brief 返回设备编号
     */
    constexpr size_type get_id() {
        return this->id_;
    }

    /**
     * @brief 唤醒该设备
     */
    virtual bool awake() = 0;

protected:
    // 设备编号
    size_type id_{};
};

/**
 * @brief CPU
 *
 */
class CPU : public _Device {
public:
    using Base = _Device;

public:
    using Base::_Device;

public:
    bool awake() override {
        return true;
    }
};

/**
 * @brief GPU
 *
 */
class GPU : public _Device {
public:
    using Self = GPU;
    using Base = _Device;
    using size_type = typename Base::size_type;
    using property_type = cudaDeviceProp;
    using property_pointer = std::shared_ptr<property_type>;
    using cublas_handle_type = cublasHandle_t;
    using cublas_handle_pointer = std::shared_ptr<cublas_handle_type>;
    using cudnn_handle_type = cudnnHandle_t;
    using cudnn_handle_pointer = std::shared_ptr<cudnn_handle_type>;

private:
    using Base::_Device;

public: // 重写构造函数
    GPU(const size_type device_id) : Base(device_id) {
        auto id = static_cast<int>(device_id);
        int device_count;
        if (cudaGetDeviceCount(&device_count)) __LLFRAME_THROW_CUDA_ERROR__
        if (id >= device_count)
            __LLFRAME_THROW_UNHANDLED_INFO__("device is not exist!")
        property_.reset(new property_type);
        if (cudaGetDeviceProperties(&(*property_), id))
            __LLFRAME_THROW_CUDA_ERROR__
        default_cudnn_handle_.reset(new cudnn_handle_type);
        if (cudnnCreate(&(*default_cudnn_handle_)))
            __LLFRAME_THROW_CUDNN_ERROR__
        default_cublas_handle_.reset(new cublas_handle_type);
        if (cublasCreate(&(*default_cublas_handle_)))
            __LLFRAME_THROW_CUBLAS_ERROR__
    }
    GPU() : GPU(0){};
    virtual ~GPU() {
        if (default_cudnn_handle_.use_count() == 1) {
            cudnnDestroy(*default_cudnn_handle_);
        }
        if (default_cublas_handle_.use_count() == 1) {
            cublasDestroy(*default_cublas_handle_);
        }
    }

public:
    bool awake() override {
        if (cudaSetDevice(this->id_)) { return false; }
        return true;
    };

    auto &cudnn_handle() {
        return *default_cudnn_handle_;
    }

    auto &cublas_handle() {
        return *default_cublas_handle_;
    }

    auto &property() {
        return *property_;
    }

protected:
    // 该设备的属性信息
    property_pointer property_{};
    // 调用cudnn默认的cudnnHandle_t
    cudnn_handle_pointer default_cudnn_handle_{};
    // 调用cublas默认的cublasHandle_t
    cublas_handle_pointer default_cublas_handle_{};
};

} // namespace llframe::device
#endif //__LLFRAME_DEVICE_IMPL_HPP__