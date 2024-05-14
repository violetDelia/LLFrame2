#include "test_config.hpp"
#ifdef TEST_MEMORY
#include "test_common.hpp"
#include <cmath>

class A {
public:
    A() {
        count++;
    }
    ~A() {
        count--;
    }
    static inline int count = 0;
};

template <llframe::device::is_Device Device>
void test_Memory_construct_and_destroy() {
    using Memory = llframe::memory::Memory<A, Device>;
    ASSERT_DEVICE_IS_VALID(Device, 0);
    IS_SAME(Device, llframe::device::GPU) {
        if constexpr (!std::is_trivial_v<A>) {
            ASSERT_THROW(Memory(1, 0), llframe::exception::Unhandled);
            return;
        }
    }
    Memory memory(5, 0);
    ASSERT_EQ(A::count, 5);
    memory.~Memory();
    ASSERT_EQ(A::count, 0);
}

template <llframe::device::is_Device Device, class Ty>
void test_Memory_defalt_construct() {
    using Memory = llframe::memory::Memory<Ty, Device>;
    ASSERT_DEVICE_IS_VALID(Device, 0);
    Memory memory;
    ASSERT_EQ(memory.get_id(), 0);
    ASSERT_EQ(memory.size(), 0);
}

template <llframe::device::is_Device Device, class Ty>
void test_Memory_construct__size_type__size_type(size_t device_id) {
    using Memory = llframe::memory::Memory<Ty, Device>;
    ASSERT_DEVICE_IS_VALID(Device, device_id);
    IS_SAME(Device, llframe::device::GPU) {
        if constexpr (!std::is_trivial_v<Ty>) {
            ASSERT_THROW(Memory(1, device_id), llframe::exception::Unhandled);
            return;
        }
    }
    for (int i = 1; i < 16; i++) {
        size_t n = (size_t{1} << i);
        Memory memory(n, device_id);
        ASSERT_EQ(memory.get_id(), device_id);
        ASSERT_EQ(memory.size(), n);
    }
}

template <llframe::device::is_Device Device, class Ty>
void test_Memory_construct_copy_and_move(size_t device_id) {
    using Memory = llframe::memory::Memory<Ty, Device>;
    ASSERT_DEVICE_IS_VALID(Device, device_id);
    IS_SAME(Device, llframe::device::GPU) {
        if constexpr (!std::is_trivial_v<Ty>) {
            ASSERT_THROW(Memory(1, device_id), llframe::exception::Unhandled);
            return;
        }
    }
    for (int i = 1; i < 10; i++) {
        size_t n = (size_t{1} << i);
        Memory memory(n, device_id);
        auto memory_copy = memory;
        auto memory_move = std::move(memory);
        ASSERT_EQ(memory.get_id(), 0);
        ASSERT_EQ(memory.size(), 0);
        ASSERT_EQ(memory_copy.get_id(), memory_move.get_id());
        ASSERT_EQ(memory_copy.size(), memory_move.size());
        for (int i = 0; i < memory_copy.size(); i++) {
            if (std::is_floating_point_v<Ty> && isnan(memory_copy.get(i)))
                continue;
            ASSERT_EQ(memory_copy.get(i), memory_move.get(i));
        }
    }
}

template <llframe::device::is_Device Device, class Ty>
void test_Memory_construct_each_type() {
    test_Memory_defalt_construct<Device, Ty>();
    for (int device_id = 0; device_id < 5; device_id++) {
        test_Memory_construct__size_type__size_type<Device, Ty>(device_id);
        test_Memory_construct_copy_and_move<Device, Ty>(device_id);
    }
}

template <llframe::device::is_Device Device>
void test_Memory_construct_each_device() {
    test_Memory_construct_and_destroy<Device>();
    APPLY_TUPLE_2(Type_Tuple, Device, test_Memory_construct_each_type);
}

TEST(Memory, construct) {
    APPLY_TUPLE(Device_Tuple, test_Memory_construct_each_device);
}

#endif // TEST_Memory