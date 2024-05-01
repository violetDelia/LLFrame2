#include "test_config.hpp"
#ifdef TEST_DEVICE
#include "test_common.hpp"
#include <cuda_runtime.h>

template <llframe::device::is_Device Device>
void test_Device_defalt_construct() {
    ASSERT_DEVICE_IS_VALID_GPU(Device, 0);
    Device device;
    ASSERT_EQ(device.get_id(), 0);
}

template <llframe::device::is_Device Device>
void test_Device_construct__size_type(size_t id) {
    ASSERT_DEVICE_IS_VALID_GPU(Device, id);
    Device device(id);
    ASSERT_EQ(device.get_id(), id);
}

template <llframe::device::is_Device Device>
void test_Device_construct_copy(size_t id) {
    ASSERT_DEVICE_IS_VALID_GPU(Device, id);
    Device device(id);
    auto device_copy = device;
    ASSERT_EQ(device.get_id(), device_copy.get_id());
}

template <llframe::device::is_Device Device>
void test_Device_construct() {
    test_Device_defalt_construct<Device>();
    test_Device_construct__size_type<Device>(0);
    test_Device_construct__size_type<Device>(1);
    test_Device_construct__size_type<Device>(2);
    test_Device_construct_copy<Device>(0);
    test_Device_construct_copy<Device>(1);
    test_Device_construct_copy<Device>(2);
}

TEST(Device, construct) {
    APPLY_TUPLE(Device_Tuple, test_Device_construct);
}

template <llframe::device::is_Device Device>
void test_Device_get_id(size_t id) {
    ASSERT_DEVICE_IS_VALID_GPU(Device, id);
    Device device(id);
    ASSERT_EQ(device.get_id(), id);
}

TEST(Device, get_id) {
    APPLY_TUPLE(Device_Tuple, test_Device_get_id, 0);
    APPLY_TUPLE(Device_Tuple, test_Device_get_id, 1);
    APPLY_TUPLE(Device_Tuple, test_Device_get_id, 2);
    APPLY_TUPLE(Device_Tuple, test_Device_get_id, 3);
}

void test_GPU_awake(size_t device_id) {
    ASSET_VALID_GPU(device_id);
    llframe::device::GPU device(device_id);
    if (device.awake()) { ASSERT_CUDA_MOLLOC_AND_MEMCPY(100); };
}

TEST(GPU, awake) {
    for (int i = 0; i < 10; i++) { test_GPU_awake(i); }
}

TEST(GPU, cublas_handle) {
}

TEST(GPU, cudnn_handle) {
}
#endif // TEST_DEVICE