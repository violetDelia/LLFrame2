#include "test_config.hpp"
#ifdef TEST_DEVICE
#include <gtest/gtest.h>
#include "test_common.hpp"
template <llframe::is_Device Device>
void test_Device_construct() {
    Device device;
    ASSERT_EQ(device.ID(), 0);
    Device device1(1);
    ASSERT_EQ(device1.ID(), 1);
    Device device2(device1);
    ASSERT_EQ(device2.ID(), 1);
    Device device3(std::move(device2));
    ASSERT_EQ(device3.ID(), 1);
    auto device4 = device3;
    ASSERT_EQ(device4.ID(), 1);
    auto device5 = std::move(device4);
    ASSERT_EQ(device5.ID(), 1);
}

template <llframe::is_Device Device>
void test_Device_ID() {
    Device device1(2);
    Device device2(device1);
    Device device3(std::move(device2));
    auto device4 = device3;
    auto device5 = std::move(device4);
    ASSERT_EQ(device1.ID(), 2);
    ASSERT_EQ(device2.ID(), 2);
    ASSERT_EQ(device3.ID(), 2);
    ASSERT_EQ(device4.ID(), 2);
    ASSERT_EQ(device5.ID(), 2);
}

TEST(Device, construct) {
    auto tuple = Device_Tuple();
    std::apply(
        [](auto &&...args) {
            (test_Device_construct<std::remove_cvref_t<decltype(args)>>(), ...);
        },
        tuple);
}
#endif // TEST_DEVICE