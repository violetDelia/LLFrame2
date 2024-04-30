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
 * @brief 设备平台实现,单例模式,设备实例从这里获取
 *
 */
#ifndef __LLFRAME_DEVICE_PLATFORM_HPP__
#define __LLFRAME_DEVICE_PLATFORM_HPP__
#include "device/device_define.hpp"
#include "device/device_impl.hpp"
#include "core/base_type.hpp"
#include <map>
#include <cuda_runtime.h>
namespace llframe { inline namespace device {
/**
 * @brief 设备平台基类
 *
 *
 *
 * @tparam Device
 */
template <is_Device Device>
class Device_Platfrom_Base {
public:
    using Self = Device_Platfrom_Base<Device>;
    using device_type = Device;
    using size_type = size_t;
    using device_map_type = std::map<size_type, device_type>;

protected:
    constexpr Device_Platfrom_Base() = default;
    constexpr Device_Platfrom_Base(const Self &other) = delete;
    constexpr Self &operator=(const Self &) = delete;

public:
    // 保存Device的容器
    device_map_type device_map;
    // 当前处于活动的设备
    int64_t active_device_id{-1};
    // 设备数量
    size_type device_nums{};
};

/**
 * @brief 不同设备初始化实现
 *
 *
 * @tparam Device
 */
template <is_Device Device>
class Device_Platfrom_Initializer : public Device_Platfrom_Base<Device> {
public:
    using Self = Device_Platfrom_Initializer<Device>;
    using Base = Device_Platfrom_Base<Device>;
    using device_type = Base::device_type;
    using size_type = Base::size_type;
    using device_map_type = Base::device_map_type;
};

template <>
class Device_Platfrom_Initializer<CPU> : public Device_Platfrom_Base<CPU> {
public:
    using Self = Device_Platfrom_Initializer<CPU>;
    using Base = Device_Platfrom_Base<CPU>;
    using device_type = Base::device_type;
    using size_type = Base::size_type;
    using device_map_type = Base::device_map_type;

protected:
    // cpu默认只有一个设备
    constexpr Device_Platfrom_Initializer() {
        this->device_map[0] = device_type(0);
        this->device_nums = 1;
    }
};

template <>
class Device_Platfrom_Initializer<GPU> : public Device_Platfrom_Base<GPU> {
public:
    using Self = Device_Platfrom_Initializer<GPU>;
    using Base = Device_Platfrom_Base<GPU>;
    using device_type = Base::device_type;
    using size_type = Base::size_type;
    using device_map_type = Base::device_map_type;

protected:
    // gpu初始化
    Device_Platfrom_Initializer() {
        int device_count;
        cudaGetDeviceCount(&device_count);
        for (int i{}; i < device_count; i++) {
            this->device_map[i] = device_type(i);
        }
        this->device_nums = device_count;
    }
};

template <is_Device Device>
class Device_Platfrom : public Device_Platfrom_Initializer<Device> {
public:
    using Self = Device_Platfrom<Device>;
    using Base = Device_Platfrom_Initializer<Device>;
    using device_type = Base::device_type;
    using size_type = Base::size_type;
    using device_map_type = Base::device_map_type;

public:
    /**
     * @brief 获取设备实例
     */
    static constexpr Self &get_instance() {
        static Self instance;
        return instance;
    };

    /**
     * @brief 唤醒指定设备
     */
    static constexpr device_type &get_device(const size_type id) {
        auto &instance = get_instance();
        if (instance.device_map.count(id)) { return instance.device_map[id]; }
        __LLFRAME_THROW_EXCEPTION_INFO__(Bad_Parameter, "device is not exist!")
    };

    /**
     * @brief 唤醒指定设备
     */
    static constexpr bool awake_device(const size_t id) {
        auto &instance = get_instance();
        if (instance.active_device_id == id) return true;
        if (get_device(id).awake()) {
            instance.active_device_id = id;
            return true;
        }
        return false;
    };
};

}}     // namespace llframe::device
#endif //__LLFRAME_DEVICE_PLATFORM_HPP__