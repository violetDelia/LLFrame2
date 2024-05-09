#include "test_config.hpp"
#ifdef TEST_MEMORY
#include "test_common.hpp"
#include <cmath>
#include <initializer_list>

template <llframe::device::is_Device Left, class Left_Ty,
          llframe::device::is_Device Right, class Right_Ty>
void test_Memory_copy_from(size_t left_id, size_t right_id, Right_Ty val) {
    using Left_Memory = llframe::memory::Memory<Left_Ty, Left>;
    using Right_Memory = llframe::memory::Memory<Right_Ty, Right>;
    ASSERT_DEVICE_IS_VALID(Left, left_id);
    ASSERT_DEVICE_IS_VALID(Right, right_id);
    IS_SAME(Left, llframe::device::GPU) {
        if constexpr (!std::is_trivial_v<Left_Ty>) {
            ASSERT_THROW(Left_Memory(1, left_id),
                         llframe::exception::Unhandled);
            return;
        }
    }
    IS_SAME(Right, llframe::device::GPU) {
        if constexpr (!std::is_trivial_v<Right_Ty>) {
            ASSERT_THROW(Left_Memory(1, right_id),
                         llframe::exception::Unhandled);
            return;
        }
    }
    Left_Memory left_memory(1, left_id);
    Right_Memory right_memory(2, right_id);
    ASSERT_THROW(left_memory.copy_form(right_memory),
                 llframe::exception::Bad_Parameter);
    for (int i = 0; i < 10; i++) {
        size_t n = (size_t{1} << i);
        Left_Memory left_memory(n, left_id);
        Right_Memory right_memory(n, right_id);
        right_memory.fill(val);
        left_memory.copy_form(right_memory);
        for (int i = 0; i < left_memory.size(); i++) {
            ASSERT_EQ(right_memory.get(i), left_memory.get(i));
        }
    }
}

template <llframe::device::is_Device Left, class Left_Ty,
          llframe::device::is_Device Right, class Right_Ty>
void test_Memory_copy_from_each_right_type() {
    for (int left_id = 0; left_id < 10; left_id++) {
        for (int right_id = 0; right_id < 10; right_id++) {
            if constexpr (llframe::is_Arithmetic<Right_Ty>
                          && llframe::is_Arithmetic<Left_Ty>) {
                test_Memory_copy_from<Left, Left_Ty, Right, Right_Ty>(
                    left_id, right_id, 127);
            }
            if constexpr (std::is_same_v<Right, std::string>
                          && std::is_same_v<Left_Ty, std::string>) {
                test_Memory_copy_from<Left, Left_Ty, Right, Right_Ty>(
                    left_id, right_id, "test");
            }
        }
    }
}

template <llframe::device::is_Device Left, class Left_Ty,
          llframe::device::is_Device Right>
void test_Memory_copy_from_each_device_right() {
    APPLY_TUPLE_4(Type_Tuple, Left, Left_Ty, Right,
                  test_Memory_copy_from_each_right_type);
}

template <llframe::device::is_Device Left, class Left_Ty>
void test_Memory_copy_from_each_type_left() {
    APPLY_TUPLE_3(Device_Tuple, Left, Left_Ty,
                  test_Memory_copy_from_each_device_right)
}

template <llframe::device::is_Device Left>
void test_Memory_copy_from_each_device_left() {
    APPLY_TUPLE_2(Type_Tuple, Left, test_Memory_copy_from_each_type_left);
}

TEST(Memory, copy_from) {
    APPLY_TUPLE(Device_Tuple, test_Memory_copy_from_each_device_left);
}

#endif // TEST_Memory