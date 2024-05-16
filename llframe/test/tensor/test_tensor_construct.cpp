#include "test_config.hpp"
#ifdef TEST_TENSOR
#include "test_common.hpp"

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

template <llframe::device::is_Device Device, size_t N>
void test_Tensor_construct_and_destroy(const llframe::shape::Shape<N> shape) {
    using Tensor = llframe::tensor::Tensor<N, A, Device>;
    ASSERT_DEVICE_IS_VALID(Device, 0);
    IS_SAME(Device, llframe::device::GPU) {
        if constexpr (!std::is_trivial_v<A>) {
            ASSERT_THROW(Tensor(shape, 0), llframe::exception::Unhandled);
            return;
        }
    }
    Tensor tensor(shape, 0);
    ASSERT_EQ(A::count, shape.count());
    tensor.~Tensor();
    ASSERT_EQ(A::count, 0);
}

template <llframe::device::is_Device Device, class Ty, size_t N>
void test_Tensor_defalt_construct() {
    using Tensor = llframe::tensor::Tensor<N, Ty, Device>;
    ASSERT_DEVICE_IS_VALID(Device, 0);
    Tensor tensor;
    ASSERT_EQ(tensor.shape(), llframe::shape::Shape<N>());
    ASSERT_EQ(tensor.memory().size(), 0);
    ASSERT_EQ(tensor.memory().get_id(), 0);
    ASSERT_EQ(tensor.get_device_id(), 0);
}

template <llframe::device::is_Device Device, class Ty, size_t N>
void test_Tensor_construct__shape_type__size_type(const llframe::shape::Shape<N> shape,
                                                  const size_t device_id) {
    using Tensor = llframe::tensor::Tensor<N, Ty, Device>;
    ASSERT_DEVICE_IS_VALID(Device, device_id);
    IS_SAME(Device, llframe::device::GPU) {
        if constexpr (!std::is_trivial_v<Ty>) {
            ASSERT_THROW(Tensor(shape, device_id), llframe::exception::Unhandled);
            return;
        }
    }
    Tensor tensor(shape, device_id);
    ASSERT_EQ(tensor.shape(), shape);
    ASSERT_EQ(tensor.memory().size(), shape.count());
    ASSERT_EQ(tensor.memory().get_id(), device_id);
    ASSERT_EQ(tensor.get_device_id(), device_id);
    ASSERT_EQ(tensor.count(), shape.count());
}

template <llframe::device::is_Device Device, class Ty, size_t N>
void test_Tensor_construct__value_type__shape_type__size_type(const llframe::shape::Shape<N> shape,
                                                              const size_t device_id, Ty val) {
    using Tensor = llframe::tensor::Tensor<N, Ty, Device>;
    ASSERT_DEVICE_IS_VALID(Device, device_id);
    IS_SAME(Device, llframe::device::GPU) {
        if constexpr (!std::is_trivial_v<Ty>) {
            ASSERT_THROW(Tensor(shape, device_id), llframe::exception::Unhandled);
            return;
        }
    }
    Tensor tensor(val, shape, device_id);
    ASSERT_EQ(tensor.shape(), shape);
    ASSERT_EQ(tensor.memory().size(), shape.count());
    ASSERT_EQ(tensor.memory().get_id(), device_id);
    ASSERT_EQ(tensor.get_device_id(), device_id);
    ASSERT_EQ(tensor.count(), shape.count());
    auto memory = tensor.memory(false);
    for (int i = 0; i < memory.size(); i++) { ASSERT_EQ(memory.get(i), val); }
}

template <llframe::device::is_Device Device, class Ty, size_t N>
void test_Tensor_construct__shape_type__init_list_type__size_type(
    const llframe::shape::Shape<N> shape, const size_t device_id,
    typename llframe::tensor::Tensor_Features<N, Ty, Device>::init_list_type init_list,
    std::initializer_list<Ty> memory_val,
    typename llframe::tensor::Tensor_Features<N, Ty, Device>::init_list_type bad_init_list) {
    using Tensor = llframe::tensor::Tensor<N, Ty, Device>;
    ASSERT_DEVICE_IS_VALID(Device, device_id);
    IS_SAME(Device, llframe::device::GPU) {
        if constexpr (!std::is_trivial_v<Ty>) {
            ASSERT_THROW(Tensor(shape, device_id), llframe::exception::Unhandled);
            return;
        }
    }
    ASSERT_THROW(Tensor( bad_init_list,shape, device_id), llframe::exception::Bad_Range);
    Tensor tensor(init_list,shape,  device_id);
    ASSERT_EQ(tensor.shape(), shape);
    ASSERT_EQ(tensor.memory().size(), shape.count());
    ASSERT_EQ(tensor.memory().get_id(), device_id);
    ASSERT_EQ(tensor.get_device_id(), device_id);
    ASSERT_EQ(tensor.count(), shape.count());
    auto it = memory_val.begin();
    auto memory = tensor.memory(false);
    for (int i = 0; i < memory.size(); i++) {
        ASSERT_EQ(memory.get(i), *it);
        it++;
    }
}

template <llframe::device::is_Device Device, class Ty, size_t N>
void test_Tensor_construct_copy_and_move(const llframe::shape::Shape<N> shape,
                                         const size_t device_id, Ty val, Ty val2) {
    if (val == val2) throw std::exception("bad parameter!");
    using Tensor = llframe::tensor::Tensor<N, Ty, Device>;
    using Shape = llframe::shape::Shape<N>;
    ASSERT_DEVICE_IS_VALID(Device, device_id);
    IS_SAME(Device, llframe::device::GPU) {
        if constexpr (!std::is_trivial_v<Ty>) {
            ASSERT_THROW(Tensor(shape, device_id), llframe::exception::Unhandled);
            return;
        }
    }
    Tensor tensor(val, shape, device_id);
    decltype(tensor) tensor_copy(tensor);
    ASSERT_EQ(tensor.shape(), tensor_copy.shape());
    ASSERT_EQ(tensor.memory().size(), tensor_copy.memory().size());
    ASSERT_EQ(tensor.memory().get_id(), tensor_copy.memory().get_id());
    ASSERT_EQ(tensor.get_device_id(), tensor_copy.get_device_id());
    ASSERT_EQ(tensor.count(), tensor_copy.count());
    auto memory = tensor.memory(false);
    auto copy_memory = tensor_copy.memory(false);
    for (int i = 0; i < memory.size(); i++) { ASSERT_EQ(memory.get(i), copy_memory.get(i)); }

    decltype(tensor) tensor_move(std::move(tensor));
    ASSERT_EQ(tensor.shape(), Shape());
    ASSERT_EQ(tensor.memory().size(), 0);
    ASSERT_EQ(tensor.memory().get_id(), 0);
    ASSERT_EQ(tensor.get_device_id(), 0);
    ASSERT_EQ(tensor.count(), 0);
    ASSERT_THROW(tensor.memory().get(0), llframe::exception::Bad_Range);
    ASSERT_EQ(tensor_move.shape(), tensor_copy.shape());
    ASSERT_EQ(tensor_move.memory().size(), tensor_copy.memory().size());
    ASSERT_EQ(tensor_move.memory().get_id(), tensor_copy.memory().get_id());
    ASSERT_EQ(tensor_move.get_device_id(), tensor_copy.get_device_id());
    ASSERT_EQ(tensor_move.count(), tensor_copy.count());
    auto move_memory = tensor_move.memory(false);
    for (int i = 0; i < move_memory.size(); i++) {
        ASSERT_EQ(move_memory.get(i), copy_memory.get(i));
    }
    for (int i = 0; i < memory.size(); i++) { memory.set(i, val2); }
    for (int i = 0; i < move_memory.size(); i++) { ASSERT_EQ(move_memory.get(i), val2); }
}

template <llframe::device::is_Device Device, class Ty, size_t N>
void test_Tensor_construct_with_shape(const llframe::shape::Shape<N> shape) {
    for (int device_id = 0; device_id < 10; device_id++) {
        test_Tensor_construct__shape_type__size_type<Device, Ty>(shape, device_id);
        if constexpr (llframe::is_Arithmetic<Ty>) {
            test_Tensor_construct__value_type__shape_type__size_type<Device, Ty>(shape, device_id,
                                                                                 5);
            test_Tensor_construct_copy_and_move<Device, Ty>(shape, device_id, 5, 4);
        }
        if constexpr (std::is_same_v<Ty, std::string>) {
            test_Tensor_construct__value_type__shape_type__size_type<Device, Ty>(shape, device_id,
                                                                                 "test");
            test_Tensor_construct_copy_and_move<Device, Ty>(shape, device_id, "test", "");
        }
    }
}

template <llframe::device::is_Device Device, class Ty>
void test_Tensor_construct_each_type() {
    test_Tensor_defalt_construct<Device, Ty, 0>();
    test_Tensor_defalt_construct<Device, Ty, 1>();
    test_Tensor_defalt_construct<Device, Ty, 2>();
    test_Tensor_defalt_construct<Device, Ty, 3>();
    test_Tensor_defalt_construct<Device, Ty, 4>();

    test_Tensor_construct_with_shape<Device, Ty>(llframe::shape::make_shape());
    test_Tensor_construct_with_shape<Device, Ty>(llframe::shape::make_shape(1));
    test_Tensor_construct_with_shape<Device, Ty>(llframe::shape::make_shape(0));
    test_Tensor_construct_with_shape<Device, Ty>(llframe::shape::make_shape(1, 2));
    test_Tensor_construct_with_shape<Device, Ty>(llframe::shape::make_shape(1, 0));
    test_Tensor_construct_with_shape<Device, Ty>(llframe::shape::make_shape(1, 2, 3));
    test_Tensor_construct_with_shape<Device, Ty>(llframe::shape::make_shape(1, 2, 0));
    test_Tensor_construct_with_shape<Device, Ty>(llframe::shape::make_shape(1, 2, 3, 4));
    test_Tensor_construct_with_shape<Device, Ty>(llframe::shape::make_shape(1, 2, 3, 0));

    for (int device_id = 0; device_id < 10; device_id++) {
        if constexpr (llframe::is_Arithmetic<Ty>) {
            test_Tensor_construct__shape_type__init_list_type__size_type<Device, Ty>(
                llframe::shape::make_shape(1), device_id, {1}, {1}, {1, 2});
            test_Tensor_construct__shape_type__init_list_type__size_type<Device, Ty>(
                llframe::shape::make_shape(1, 2), device_id, {{1}}, {1, 0}, {{1, 2, 3}});
            test_Tensor_construct__shape_type__init_list_type__size_type<Device, Ty>(
                llframe::shape::make_shape(1, 2, 3), device_id, {{{}, {1}}}, {0, 0, 0, 1, 0, 0},
                {{{}, {1, 2, 3, 4}}});
        }
        if constexpr (std::is_same_v<Ty, std::string>) {
            test_Tensor_construct__shape_type__init_list_type__size_type<Device, Ty>(
                llframe::shape::make_shape(1), device_id, {"1"}, {"1"}, {"1", "2"});
            test_Tensor_construct__shape_type__init_list_type__size_type<Device, Ty>(
                llframe::shape::make_shape(1, 2), device_id, {{"1"}}, {"1", ""}, {{"1", "2","3"}});
            test_Tensor_construct__shape_type__init_list_type__size_type<Device, Ty>(
                llframe::shape::make_shape(1, 2, 3), device_id, {{{"2"}, {"1"}}},
                {"2", "", "", "1", "", ""}, {{{"2"}, {"1", "2", "3", "4"}}});
        }
    }
}

template <llframe::device::is_Device Device>
void test_Tensor_construct_each_device() {
    test_Tensor_construct_and_destroy<Device>(llframe::shape::make_shape());
    test_Tensor_construct_and_destroy<Device>(llframe::shape::make_shape(1));
    test_Tensor_construct_and_destroy<Device>(llframe::shape::make_shape(1, 2));
    test_Tensor_construct_and_destroy<Device>(llframe::shape::make_shape(1, 2, 3));
    test_Tensor_construct_and_destroy<Device>(llframe::shape::make_shape(1, 2, 3, 4));
    APPLY_TUPLE_2(Type_Tuple, Device, test_Tensor_construct_each_type);
}

TEST(Tensor, construct) {
    APPLY_TUPLE(Device_Tuple, test_Tensor_construct_each_device);
}

#endif // TEST_TENSOR