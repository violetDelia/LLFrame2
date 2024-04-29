#include "test_config.hpp"
#ifdef TEST_SHAPE
#include <gtest/gtest.h>
#include "test_common.hpp"
#include "core/shape.hpp"

template <size_t N_Dim>
void test_Shape_defalt_construct() {
    llframe::Shape<N_Dim> shape;
    for (int i = 0; i < N_Dim; i++) { ASSERT_EQ(shape[i], 0); }
    ASSERT_EQ(shape.size(), N_Dim);
    ASSERT_EQ(shape.dims(), N_Dim);
}
template <class T, T... ints>
void test_Shape_construct_intergrals_impl(
    std::integer_sequence<T, ints...> seq) {
    llframe::Shape<sizeof...(ints)> shape(ints...);
    ASSERT_EQ(shape.size(), sizeof...(ints));
    ASSERT_EQ(shape.dims(), sizeof...(ints));
    for (int i = 0; i < shape.dims(); i++) { ASSERT_EQ(shape[i], i); }
    if constexpr (sizeof...(ints) > 1) {
        llframe::Shape<sizeof...(ints) + 1> shape_incomplet(ints...);
        ASSERT_EQ(shape_incomplet.size(), sizeof...(ints) + 1);
        ASSERT_EQ(shape_incomplet.dims(), sizeof...(ints) + 1);
        for (int i = 0; i < shape.dims() - 1; i++) {
            ASSERT_EQ(shape_incomplet[i], i);
        }
        ASSERT_EQ(shape_incomplet[shape_incomplet.dims() - 1], 0);
    }
}

template <size_t N_Dim>
void test_Shape_construct_intergrals() {
    test_Shape_construct_intergrals_impl(std::make_index_sequence<N_Dim>());
}

template <llframe::is_Integral... Integrals>
void test_Shape_make_shape(Integrals... values) {
    auto shape = llframe::make_shape(values...);
    ASSERT_EQ(shape.size(), sizeof...(Integrals));
    std::array<typename decltype(shape)::value_type, sizeof...(Integrals)> arr{
        values...};
    for (int i = 0; i < shape.dims(); i++) { ASSERT_EQ(shape[i], arr[i]); }
}

template <llframe::is_Integral... Integrals>
void test_Shape_construct_copy_and_move(Integrals... values) {
    auto shape = llframe::make_shape(values...);
    auto shape_copy(shape);
    for (int i = 0; i < shape.dims(); i++) {
        ASSERT_EQ(shape[i], shape_copy[i]);
    }
    ASSERT_EQ(shape.size(), shape_copy.size());
    ASSERT_EQ(shape.dims(), shape_copy.dims());
    auto shape_move(std::move(shape_copy));
    for (int i = 0; i < shape.dims(); i++) {
        ASSERT_EQ(shape[i], shape_move[i]);
        ASSERT_EQ(shape_copy[i], 0);
    }
    ASSERT_EQ(shape_move.size(), shape_copy.size());
    ASSERT_EQ(shape_move.dims(), shape_copy.dims());
}

template <llframe::is_Integral... Integrals>
void test_Shape_operator_assign(Integrals... values) {
    auto shape = llframe::make_shape(values...);
    decltype(shape) shape_copy;
    shape_copy = shape;
    for (int i = 0; i < shape.dims(); i++) {
        ASSERT_EQ(shape[i], shape_copy[i]);
    }
    ASSERT_EQ(shape.size(), shape_copy.size());
    ASSERT_EQ(shape.dims(), shape_copy.dims());
    decltype(shape) shape_move;
    shape_move = std::move(shape_copy);
    for (int i = 0; i < shape.dims(); i++) {
        ASSERT_EQ(shape[i], shape_move[i]);
        ASSERT_EQ(shape_copy[i], 0);
    }
    ASSERT_EQ(shape_move.size(), shape_copy.size());
    ASSERT_EQ(shape_move.dims(), shape_copy.dims());
}

template <llframe::is_Integral... Integrals>
void test_Shape_construct_iterator(Integrals... values) {
    std::array<int, sizeof...(Integrals)> arr{values...};
    llframe::Shape<sizeof...(Integrals)> shape(arr.begin(), arr.end());
    for (int i = 0; i < shape.dims(); i++) { ASSERT_EQ(shape[i], arr[i]); }
    ASSERT_THROW(
        llframe::Shape<sizeof...(Integrals)>(arr.begin(), arr.end() + 1),
        llframe::Bad_Range);
    if constexpr (sizeof...(Integrals) > 0) {
        ASSERT_THROW(
            llframe::Shape<sizeof...(Integrals)>(arr.end(), arr.begin()),
            llframe::Bad_Range);
        llframe::Shape<sizeof...(Integrals)> shape_incomplete(arr.begin(),
                                                              arr.end() - 1);
        for (int i = 0; i < shape_incomplete.dims() - 1; i++) {
            ASSERT_EQ(shape_incomplete[i], arr[i]);
        }
        ASSERT_EQ(shape_incomplete[shape_incomplete.size() - 1], 0);
    }
}

template <llframe::is_Integral... Integrals>
void test_Shape_construct(Integrals... values) {
    test_Shape_defalt_construct<sizeof...(Integrals)>();
    test_Shape_construct_intergrals<sizeof...(Integrals)>();
    test_Shape_make_shape(values...);
    test_Shape_construct_copy_and_move(values...);
    test_Shape_operator_assign(values...);
    test_Shape_construct_iterator(values...);
}

TEST(Shape, construct) {
    test_Shape_construct();
    test_Shape_construct(1);
    test_Shape_construct(1, 2);
    test_Shape_construct(1, 2, 3);
    test_Shape_construct(1, 2, 3, 4);
}

template <llframe::is_Integral... Integrals>
void test_Shape_operator_array_subscript(Integrals... values) {
    auto shape = llframe::make_shape(values...);
    auto shape_copy(shape);
    for (size_t i{}; i < shape.dims(); i++) {
        shape_copy[i] += 1;
        ASSERT_EQ(shape_copy[i], shape[i] + 1);
    }
    ASSERT_THROW(auto placeholder = shape[shape.dims()], std::out_of_range);
    ASSERT_THROW(auto placeholder = shape_copy[shape_copy.dims()],
                 std::out_of_range);
}
TEST(Shape, operator_array_subscript) {
    test_Shape_operator_array_subscript();
    test_Shape_operator_array_subscript(1);
    test_Shape_operator_array_subscript(1, 2);
    test_Shape_operator_array_subscript(1, 2, 3);
    test_Shape_operator_array_subscript(1, 2, 3, 4);
}

template <llframe::is_Integral... Integrals>
void test_Shape_count(std::int64_t count, Integrals... values) {
    auto shape = llframe::make_shape(values...);
    ASSERT_EQ(shape.count(), count);
}

TEST(Shape, count) {
    test_Shape_count(0);
    test_Shape_count(1, 1);
    test_Shape_count(1 * (-2), 1, -2);
    test_Shape_count(1 * (-2) * 3, 1, -2, 3);
    test_Shape_count(1 * (-2) * 3 * (-4), 1, -2, 3, -4);
}

template <llframe::is_Integral... Integrals>
void test_Shape_operator_equal(Integrals... values) {
    llframe::Shape<sizeof...(Integrals)> shape1(values...);
    llframe::Shape<sizeof...(Integrals)> shape2(values...);
    ASSERT_TRUE(shape1 == shape2);
    ASSERT_FALSE(shape1 != shape2);
    llframe::Shape<sizeof...(Integrals) + 1> shape3(values...);
    ASSERT_TRUE(shape1 != shape3);
    ASSERT_FALSE(shape1 == shape3);
    if constexpr (sizeof...(Integrals) > 1) {
        shape1[0] += 1;
        ASSERT_TRUE(shape1 != shape2);
        ASSERT_FALSE(shape1 == shape2);
    }
}

TEST(Shape, operator_equal) {
    test_Shape_operator_equal();
    test_Shape_operator_equal(1);
    test_Shape_operator_equal(1, 2);
    test_Shape_operator_equal(1, 2, 3);
    test_Shape_operator_equal(1, 2, 3, 4);
}

#endif // TEST_SHAPE