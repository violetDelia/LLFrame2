#include "test_config.hpp"
#ifdef TEST_TENSOR
#include "test_common.hpp"

template <llframe::device::is_Device Device, class Ty, size_t N>
void test_Tensor_count(const llframe::shape::Shape<N> shape, const size_t device_id) {
    using Tensor = llframe::tensor::Tensor<N, Ty, Device>;
    ASSERT_DEVICE_IS_VALID(Device, device_id);
    IS_SAME(Device, llframe::device::GPU) {
        if constexpr (!std::is_trivial_v<Ty>) {
            ASSERT_THROW(Tensor(shape, device_id), llframe::exception::Unhandled);
            return;
        }
    }
    Tensor tensor(shape, device_id);
    ASSERT_EQ(tensor.count(), shape.count());
}

template <llframe::device::is_Device Device, class Ty, size_t N>
void test_Tensor_count_with_shape(const llframe::shape::Shape<N> shape) {
    for (int device_id = 0; device_id < 10; device_id++) {
        test_Tensor_count<Device, Ty>(shape, device_id);
        if constexpr (llframe::is_Arithmetic<Ty>) {}
        if constexpr (std::is_same_v<Ty, std::string>) {}
    }
}

template <llframe::device::is_Device Device, class Ty>
void test_Tensor_count_each_type() {
    test_Tensor_count_with_shape<Device, Ty>(llframe::shape::make_shape());
    test_Tensor_count_with_shape<Device, Ty>(llframe::shape::make_shape(1));
    test_Tensor_count_with_shape<Device, Ty>(llframe::shape::make_shape(0));
    test_Tensor_count_with_shape<Device, Ty>(llframe::shape::make_shape(1, 2));
    test_Tensor_count_with_shape<Device, Ty>(llframe::shape::make_shape(1, 0));
    test_Tensor_count_with_shape<Device, Ty>(llframe::shape::make_shape(1, 2, 3));
    test_Tensor_count_with_shape<Device, Ty>(llframe::shape::make_shape(1, 2, 0));
    test_Tensor_count_with_shape<Device, Ty>(llframe::shape::make_shape(1, 2, 3, 4));
    test_Tensor_count_with_shape<Device, Ty>(llframe::shape::make_shape(1, 2, 3, 0));
}

template <llframe::device::is_Device Device>
void test_Tensor_count_each_device() {
    APPLY_TUPLE_2(Type_Tuple, Device, test_Tensor_count_each_type);
}

TEST(Tensor, count) {
    APPLY_TUPLE(Device_Tuple, test_Tensor_count_each_device);
}

#endif // TEST_TENSOR