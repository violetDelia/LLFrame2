#include "test_config.hpp"
#ifdef TEST_MEMORY
#include "test_common.hpp"
#include <cmath>

template <llframe::device::is_Device Device, class Ty>
void test_Memory_use_count(size_t device_id, Ty val, Ty val2) {
    using Memory = llframe::memory::Memory<Ty, Device>;
    ASSERT_DEVICE_IS_VALID(Device, device_id);
    IS_SAME(Device, llframe::device::GPU) {
        if constexpr (!std::is_trivial_v<Ty>) {
            ASSERT_THROW(Memory(1, device_id), llframe::exception::Unhandled);
            return;
        }
    }
    for (int i = 0; i < 10; i++) {
        size_t n = (size_t{1} << i);
        Memory memory(n, device_id);
        memory.fill(val);
        auto memory_ref = memory.ref();
        ASSERT_EQ(memory.use_count(), 2);
        ASSERT_EQ(memory_ref.use_count(), 2);
    }
}

template <llframe::device::is_Device Device, class Ty>
void test_Memory_use_count_each_type() {
    for (int i = 0; i < 10; i++) {
        if constexpr (llframe::is_Arithmetic<Ty>) {
            test_Memory_use_count<Device, Ty>(i, 127, 126);
        }
        if constexpr (std::is_same_v<Ty, std::string>) {
            test_Memory_use_count<Device, Ty>(i, "test", " ");
        }
    }
}

template <llframe::device::is_Device Device>
void test_Memory_use_count_each_device() {
    APPLY_TUPLE_2(Type_Tuple, Device, test_Memory_use_count_each_type);
}

TEST(Memory, use_count) {
    APPLY_TUPLE(Device_Tuple, test_Memory_use_count_each_device);
}

#endif // TEST_Memory