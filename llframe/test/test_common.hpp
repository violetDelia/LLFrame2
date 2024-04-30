
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

#define IS_SAME(Ty1, Ty2) if constexpr (std::is_same_v<Ty1, Ty2>)
#define IS_NOT_SAME(Ty1, Ty2) if constexpr (!std::is_same_v<Ty1, Ty2>)
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

#define TEST_CUDA_MOLLOC_AND_MEMCPY(value)                                     \
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
#define ASSET_VALID_GPU(id)                                                    \
    int device_num;                                                            \
    if (cudaGetDeviceCount(&device_num)) return;                               \
    if (id >= device_num) return;

#define ASSERT_DEVICE_IS_VALID_GPU(Device, id)                                 \
    IS_SAME(Device, llframe::device::GPU) {                                    \
        ASSET_VALID_GPU(id)                                                    \
        return;                                                                \
    }
#endif //__LLFRAME_TEST_COMMON__