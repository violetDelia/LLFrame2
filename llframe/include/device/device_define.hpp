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
 * @brief device 定义
 *
 */
#ifndef LLFRAME_DEVICE_DEVICE_DEFINE_HPP
#define LLFRAME_DEVICE_DEVICE_DEFINE_HPP
#include <type_traits>
namespace llframe::device {

class CPU;
class GPU;

/**
 * @brief 判断类型是否是Device的基类
 */
template <class Ty>
concept is_Device = std::is_same_v<CPU, Ty> || std::is_same_v<GPU, Ty>;

template <is_Device Device>
class Device_Platform;

} // namespace llframe::device
#endif // LLFRAME_DEVICE_DEVICE_DEFINE_HPP