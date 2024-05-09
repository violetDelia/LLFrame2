#include "test_config.hpp"
#ifdef TEST_MEMORY
#include "test_common.hpp"
#include <cmath>

template <llframe::device::is_Device Device, class Ty>
void test_Memory_set(size_t device_id, Ty val) {
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
        for (int i = 0; i < memory.size(); i++) {
            memory.set(i, val);
            ASSERT_EQ(memory.get(i), val);
        }
    }
}

template <llframe::device::is_Device Device, class Ty>
void test_Memory_set_each_type() {
    for (int i = 0; i < 10; i++) {
        if constexpr (llframe::is_Arithmetic<Ty>) {
            test_Memory_set<Device, Ty>(i, 127);
        }
        if constexpr (std::is_same_v<Ty, std::string>) {
            test_Memory_set<Device, Ty>(i, "test");
        }
    }
}

template <llframe::device::is_Device Device>
void test_Memory_set_each_device() {
    APPLY_TUPLE_2(Type_Tuple, Device, test_Memory_set_each_type);
}

TEST(Memory, set) {
    APPLY_TUPLE(Device_Tuple, test_Memory_set_each_device);
}

#endif // TEST_Memory