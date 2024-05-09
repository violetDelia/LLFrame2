#include "test_config.hpp"
#ifdef TEST_MEMORY
#include "test_common.hpp"
#include <cmath>

template <llframe::device::is_Device Device, class Ty>
void test_Memory_fill(size_t device_id, Ty val,
                      std::initializer_list<Ty> init_list) {
    using Memory = llframe::memory::Memory<Ty, Device>;
    ASSERT_DEVICE_IS_VALID(Device, device_id);
    IS_SAME(Device, llframe::device::GPU) {
        if constexpr (!std::is_trivial_v<Ty>) {
            ASSERT_THROW(Memory(1, device_id), llframe::exception::Unhandled);
            return;
        }
    }
    for (int i = 0; i < 10; i++) {
        for (int k = 0; k < 10; k++) {
            size_t n = (size_t{1} << i);
            size_t pos = (size_t{1} << k);
            Memory memory(n, device_id);
            memory.fill(val);
            if (pos + init_list.size() > n) {
                ASSERT_THROW(memory.fill(pos, init_list),
                             llframe::exception::Bad_Range);
                continue;
            }
            memory.fill(pos, init_list);
            for (int i = 0; i < memory.size(); i++) {
                if (i >= pos && i < pos + init_list.size()) {
                    ASSERT_EQ(memory.get(i), *(init_list.begin() + i - pos));
                } else {
                    ASSERT_EQ(memory.get(i), val);
                }
            }
        }
    }
}

template <llframe::device::is_Device Device, class Ty>
void test_Memory_fill(size_t device_id, Ty val, Ty val2) {
    using Memory = llframe::memory::Memory<Ty, Device>;
    ASSERT_DEVICE_IS_VALID(Device, device_id);
    IS_SAME(Device, llframe::device::GPU) {
        if constexpr (!std::is_trivial_v<Ty>) {
            ASSERT_THROW(Memory(1, device_id), llframe::exception::Unhandled);
            return;
        }
    }
    for (int i = 0; i < 10; i++) {
        for (int k = 0; k < 10; k++) {
            for (int j = 0; j < 10; j++) {
                size_t n = (size_t{1} << i);
                size_t pos = (size_t{1} << k);
                size_t n_pos = (size_t{1} << j);
                Memory memory(n, device_id);
                memory.fill(val);
                if (pos + n_pos > n) {
                    ASSERT_THROW(memory.fill(pos, n_pos, val2),
                                 llframe::exception::Bad_Range);
                    continue;
                }
                memory.fill(pos, n_pos, val2);
                for (int i = 0; i < memory.size(); i++) {
                    if (i >= pos && i < pos + n_pos) {
                        ASSERT_EQ(memory.get(i), val2);
                    } else {
                        ASSERT_EQ(memory.get(i), val);
                    }
                }
            }
        }
    }
}

template <llframe::device::is_Device Device, class Ty>
void test_Memory_fill_each_type() {
    for (int i = 0; i < 10; i++) {
        if constexpr (llframe::is_Arithmetic<Ty>) {
            test_Memory_fill<Device, Ty>(i, 127, 126);
            test_Memory_fill<Device, Ty>(i, 127, {});
            test_Memory_fill<Device, Ty>(i, 127, {1, 2, 3, 4, 5, 6});
        }
        if constexpr (std::is_same_v<Ty, std::string>) {
            test_Memory_fill<Device, Ty>(i, "test", " ");
            test_Memory_fill<Device, Ty>(i, "test", {});
            test_Memory_fill<Device, Ty>(i, "test", {"1", "2", "3", "4"});
        }
    }
}

template <llframe::device::is_Device Device>
void test_Memory_fill_each_device() {
    APPLY_TUPLE_2(Type_Tuple, Device, test_Memory_fill_each_type);
}

TEST(Memory, fill) {
    APPLY_TUPLE(Device_Tuple, test_Memory_fill_each_device);
}

#endif // TEST_Memory