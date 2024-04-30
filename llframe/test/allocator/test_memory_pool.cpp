#include "test_config.hpp"
#ifdef TEST_ALLOCATOR
#include "test_common.hpp"
#include "allocator/allocator.hpp"

template <class Allocator>
void test_memory_pool_each_bytes(size_t bytes) {
    Allocator::config::reset(bytes);
    auto placeholder1 = Allocator::allocate(1);
    auto placeholder2 = Allocator::allocate(1);
    auto placeholder3 = Allocator::allocate(1);
}

template <class Ty, llframe::device::is_Device Device>
void test_Memory_Pool(size_t device_id) {
    using allocator = llframe::allocator::Allocator<Ty, Device>;
    ASSERT_DEVICE_IS_VALID(Device, device_id);

    for (int bytes = 128; bytes < 1048576; bytes = bytes * 2) {
        test_memory_pool_each_bytes<allocator>(bytes);
        ASSERT_EQ(allocator::memory_pool::get_instance(device_id)[bytes].size(),
                  3);
    }
}

template <class Ty>
void test_Memory_Pool_for_each_device() {
    APPLY_TUPLE_2(Device_Tuple, Ty, test_Memory_Pool, 0);
    APPLY_TUPLE_2(Device_Tuple, Ty, test_Memory_Pool, 1);
    APPLY_TUPLE_2(Device_Tuple, Ty, test_Memory_Pool, 2);
    APPLY_TUPLE_2(Device_Tuple, Ty, test_Memory_Pool, 3);
    APPLY_TUPLE_2(Device_Tuple, Ty, test_Memory_Pool, 4);
    APPLY_TUPLE_2(Device_Tuple, Ty, test_Memory_Pool, 5);
}

TEST(Memory_Pool, all) {
    APPLY_TUPLE(Type_Tuple, test_Memory_Pool_for_each_device);
}
#endif // TEST_ALLOCATOR