#include "test_config.hpp"
#ifdef TEST_ALLOCATOR
#include "test_common.hpp"
#include "allocator/allocator.hpp"

template <class Allocator>
void test_memory_pool() {
    Allocator::config.reset(2048);
    auto placeholder1 = Allocator::allocate(5);
    auto placeholder2 = Allocator::allocate(0);
    auto placeholder3 = Allocator::allocate(3);
    auto placeholder4 = Allocator::allocate(2);
}

template <class Ty, llframe::device::is_Device Device>
void test_Allocator(size_t device_id) {
    using allocator = llframe::allocator::Allocator<Ty, Device>;
    test_allocator_traits<allocator>();
    if (sizeof(Ty) != 1) {
        ASSERT_THROW(auto p = allocator::allocate(
                         std::numeric_limits<size_t>::max(), device_id),
                     llframe::exception::Bad_Alloc);
    };
    ASSERT_EQ(allocator::allocate(0, device_id),
              typename allocator::shared_pointer(nullptr));
    ASSERT_DEVICE_IS_VALID(Device, device_id);
    for (int i = 0; i < 100; i++) {
        auto p = allocator::allocate(i, device_id);
    }
}

template <class Ty>
void test_Allocator_for_each_device() {
    APPLY_TUPLE_2(Device_Tuple, Ty, test_Allocator, 0);
    APPLY_TUPLE_2(Device_Tuple, Ty, test_Allocator, 1);
    APPLY_TUPLE_2(Device_Tuple, Ty, test_Allocator, 2);
    APPLY_TUPLE_2(Device_Tuple, Ty, test_Allocator, 3);
    APPLY_TUPLE_2(Device_Tuple, Ty, test_Allocator, 4);
    APPLY_TUPLE_2(Device_Tuple, Ty, test_Allocator, 5);
}

TEST(Allocator, all) {
    APPLY_TUPLE(Type_Tuple, test_Allocator_for_each_device);
}
#endif // TEST_ALLOCATOR