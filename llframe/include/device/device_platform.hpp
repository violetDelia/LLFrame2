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
namespace llframe ::device {
/**
 * @brief 设备平台基类
 *
 *
 *
 * @tparam Device
 */
template <is_Device Device>
class _Device_Platform_Base {
public:
    using Self = _Device_Platform_Base<Device>;
    using device_type = Device;
    using size_type = size_t;
    using device_map_type = std::map<size_type, device_type>;

public:
    // 构造函数是不会发生异常
    static constinit const bool construct_no_thorw = true;

protected:
    constexpr _Device_Platform_Base() noexcept = default;
    constexpr _Device_Platform_Base(const Self &other) noexcept = delete;
    constexpr Self &operator=(const Self &) noexcept = delete;

public:
    // 保存Device的容器
    device_map_type device_map;
    // 当前处于活动的设备
    int64_t active_device_id{-1};
    // 设备数量
    size_type device_nums{};
};

/**
 * @brief
 * 不同设备平台初始化实现的中间类,负责设备平台的初始化,只实现对应平台的构造函数
 */
template <is_Device Device>
class _Device_Platform_Initializer : public _Device_Platform_Base<Device> {
public:
    using Self = _Device_Platform_Initializer<Device>;
    using Base = _Device_Platform_Base<Device>;
    using device_type = typename Base::device_type;
    using size_type = typename Base::size_type;
    using device_map_type = typename Base::device_map_type;

public:
    // 构造函数是不会发生异常
    static constinit const bool construct_no_thorw = Base::construct_no_thorw;
};

template <>
class _Device_Platform_Initializer<CPU> : public _Device_Platform_Base<CPU> {
public:
    using Self = _Device_Platform_Initializer<CPU>;
    using Base = _Device_Platform_Base<CPU>;
    using device_type = typename Base::device_type;
    using size_type = typename Base::size_type;
    using device_map_type = typename Base::device_map_type;

public:
    // 构造函数是不会发生异常
    static constinit const bool construct_no_thorw =
        device_type::construct_no_thorw;

protected:
    // cpu默认只有一个设备
    constexpr _Device_Platform_Initializer() noexcept(construct_no_thorw) {
        this->device_map[0] = device_type(0);
        this->device_nums = 1;
    }
};

template <>
class _Device_Platform_Initializer<GPU> : public _Device_Platform_Base<GPU> {
public:
    using Self = _Device_Platform_Initializer<GPU>;
    using Base = _Device_Platform_Base<GPU>;
    using device_type = typename Base::device_type;
    using size_type = typename Base::size_type;
    using device_map_type = typename Base::device_map_type;

public:
    // 构造函数是不会发生异常
    static constinit const bool construct_no_thorw =
        device_type::construct_no_thorw;

protected:
    // gpu初始化
    _Device_Platform_Initializer() noexcept(construct_no_thorw) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        for (int i{}; i < device_count; i++) {
            this->device_map[i] = device_type(i);
        }
        this->device_nums = device_count;
        if (this->device_nums > 0) {
            if (!this->device_map[0].awake()) {
                __LLFRAME_THROW_UNHANDLED_INFO__("awake device falut!")
            };
            this->active_device_id = 0;
        }
    }
};

/**
 * @brief 设备平台,对外开放的查询和修改设备信息的接口
 */
template <is_Device Device>
class Device_Platform : public _Device_Platform_Initializer<Device> {
public:
    using Self = Device_Platform<Device>;
    using Base = _Device_Platform_Initializer<Device>;
    using device_type = typename Base::device_type;
    using size_type = typename Base::size_type;
    using device_map_type = typename Base::device_map_type;

public:
    // 构造函数是不会发生异常
    static constinit const bool construct_no_thorw = Base::construct_no_thorw;

public:
    /**
     * @brief 获取设备实例
     */
    static constexpr Self &get_instance() noexcept(construct_no_thorw) {
        static Self instance;
        return instance;
    };

    /**
     * @brief 获取指定设备
     */
    static constexpr device_type &
    get_device(const size_type id) noexcept(construct_no_thorw) {
        auto &instance = get_instance();
        if (instance.device_map.count(id)) { return instance.device_map[id]; }
        __LLFRAME_THROW_EXCEPTION_INFO__(exception::Bad_Parameter,
                                         "device is not exist!")
    };

    /**
     * @brief 唤醒指定设备
     */
    static constexpr bool
    awake_device(const size_type id) noexcept(construct_no_thorw) {
        auto &instance = get_instance();
        if (instance.active_device_id == id) return true;
        if (get_device(id).awake()) {
            instance.active_device_id = id;
            return true;
        }
        return false;
    };

    /**
     * @brief 获取当前活动的设备
     */
    static constexpr device_type &
    get_active_device() noexcept(construct_no_thorw) {
        auto &instance = get_instance();
        return instance.device_map[instance.active_device_id];
    };
};

} // namespace llframe::device
#endif //__LLFRAME_DEVICE_PLATFORM_HPP__