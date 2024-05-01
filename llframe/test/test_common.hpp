
#ifndef __LLFRAME_TEST_COMMON__
#define __LLFRAME_TEST_COMMON__
#include <gtest/gtest.h>
#include "core/base_type.hpp"
#include "core/exception.hpp"
#include "device/device_define.hpp"
#include "device/device_impl.hpp"
#include <cuda_runtime.h>
using Exception_Tuple =
    std::tuple<llframe::exception::Bad_Alloc, llframe::exception::Bad_Parameter,
               llframe::exception::Bad_Index, llframe::exception::Exception,
               llframe::exception::Null_Pointer,
               llframe::exception::STD_Exception, llframe::exception::Unhandled,
               llframe::exception::Unimplement, llframe::exception::Unknown,
               llframe::exception::Bad_Range, llframe::exception::CUDA_Error,
               llframe::exception::CuBLAS_Errot,
               llframe::exception::CuDNN_Errot>;

using Device_Tuple = std::tuple<llframe::device::CPU, llframe::device::GPU>;
using Type_Tuple = std::tuple<int, uint16_t, float, double, std::string>;

// 类型相同
#define IS_SAME(Ty1, Ty2) if constexpr (std::is_same_v<Ty1, Ty2>)
// 类型不同
#define IS_NOT_SAME(Ty1, Ty2) if constexpr (!std::is_same_v<Ty1, Ty2>)
// apply宏
#define APPLY_TUPLE(Tuple, func, ...)                                          \
    {                                                                          \
        auto tuple = Tuple();                                                  \
        std::apply(                                                            \
            [](auto &&...args) {                                               \
                (func<std::remove_cvref_t<decltype(args)>>(##__VA_ARGS__),     \
                 ...);                                                         \
            },                                                                 \
            tuple);                                                            \
    };
// apply宏2
#define APPLY_TUPLE_2(Tuple, Tp, func, ...)                                    \
    {                                                                          \
        auto tuple = Tuple();                                                  \
        std::apply(                                                            \
            [](auto &&...args) {                                               \
                (func<Tp, std::remove_cvref_t<decltype(args)>>(##__VA_ARGS__), \
                 ...);                                                         \
            },                                                                 \
            tuple);                                                            \
    };

// 确保可以使用cuda分配内存
#define ASSERT_CUDA_MOLLOC_AND_MEMCPY(value)                                   \
    {                                                                          \
        decltype(value) *device_p{};                                           \
        decltype(value) host_p[1];                                             \
        decltype(value) temp_p[1];                                             \
        *host_p = value;                                                       \
        auto size = sizeof(decltype(*host_p));                                 \
        cudaMalloc(&device_p, size);                                           \
        cudaMemcpy(device_p, host_p, size, cudaMemcpyHostToDevice);            \
        cudaMemcpy(temp_p, device_p, size, cudaMemcpyDeviceToHost);            \
        ASSERT_EQ(*temp_p, *host_p);                                           \
    }
// id对GPU是有效的
#define ASSET_VALID_GPU(id)                                                    \
    int device_num;                                                            \
    if (cudaGetDeviceCount(&device_num)) return;                               \
    if (id >= device_num) return;
// 确保设备是有效的GPU,如果Device是GPU
#define ASSERT_DEVICE_IS_VALID_GPU(Device, id)                                 \
    IS_SAME(Device, llframe::device::GPU) {                                    \
        ASSET_VALID_GPU(id)                                                    \
    }
// id对cpu是有效的
#define ASSET_VALID_CPU(id)                                                    \
    if (id >= 1) return;
// 确保设备是有效的CPU,如果Device是CPU
#define ASSERT_DEVICE_IS_VALID_CPU(Device, id)                                 \
    IS_SAME(Device, llframe::device::CPU) {                                    \
        ASSET_VALID_CPU(id)                                                    \
    }
// 确保设备是有效的
#define ASSERT_DEVICE_IS_VALID(Device, id)                                     \
    ASSERT_DEVICE_IS_VALID_GPU(Device, id)                                     \
    ASSERT_DEVICE_IS_VALID_CPU(Device, id)
// 测试Allocator的属性
template <class Allocator>
void test_allocator_traits() {
    ASSERT_EQ((std::is_same_v<
                  typename std::allocator_traits<Allocator>::const_pointer,
                  typename Allocator::const_pointer>),
              true);
    ASSERT_EQ((std::is_same_v<
                  typename std::allocator_traits<Allocator>::const_void_pointer,
                  const void *>),
              true);
    ASSERT_EQ((std::is_same_v<
                  typename std::allocator_traits<Allocator>::difference_type,
                  typename Allocator::difference_type>),
              true);
    ASSERT_EQ(
        (std::is_same_v<typename std::allocator_traits<Allocator>::pointer,
                        typename Allocator::pointer>),
        true);
    ASSERT_EQ(
        (std::is_same_v<typename std::allocator_traits<
                            Allocator>::propagate_on_container_copy_assignment,
                        std::false_type>),
        true);
    ASSERT_EQ(
        (std::is_same_v<typename std::allocator_traits<
                            Allocator>::propagate_on_container_move_assignment,
                        std::false_type>),
        true);
    ASSERT_EQ((std::is_same_v<typename std::allocator_traits<
                                  Allocator>::propagate_on_container_swap,
                              std::false_type>),
              true);
    ASSERT_EQ(
        (std::is_same_v<typename std::allocator_traits<Allocator>::size_type,
                        typename Allocator::size_type>),
        true);
    ASSERT_EQ(
        (std::is_same_v<typename std::allocator_traits<Allocator>::value_type,
                        typename Allocator::value_type>),
        true);
    ASSERT_EQ(
        (std::is_same_v<typename std::allocator_traits<Allocator>::void_pointer,
                        void *>),
        true);
};
#endif //__LLFRAME_TEST_COMMON__