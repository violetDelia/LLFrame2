#include "test_config.hpp"
#ifdef TEST_EXCEPTION
#include <gtest/gtest.h>
#include "test_common.hpp"

template <class Exception>
void test_Exception_construct() {
    Exception exception1;
    Exception exception2("exception");
    ASSERT_EQ(exception2.what(), "exception\n");
    Exception exception3("exception", "file", 1, "func_name");
    ASSERT_EQ(exception3.what(), "exception\n\tfunc_name: file<1>\n");
    Exception exception4("file", 1, "func_name");
    Exception exception5(exception3);
    ASSERT_EQ(exception5.what(), "exception\n\tfunc_name: file<1>\n");
    Exception exception6(std::move(exception5));
    ASSERT_EQ(exception6.what(), "exception\n\tfunc_name: file<1>\n");
    ASSERT_EQ(exception5.what(), "");
    auto exception7 = exception6;
    ASSERT_EQ(exception7.what(), "exception\n\tfunc_name: file<1>\n");
    auto exception8 = std::move(exception7);
    ASSERT_EQ(exception7.what(), "");
    ASSERT_EQ(exception8.what(), "exception\n\tfunc_name: file<1>\n");
}

TEST(Exception, construct) {
    auto tuple = Exception_Tuple();
    std::apply(
        [](auto &&...args) {
            (test_Exception_construct<std::remove_cvref_t<decltype(args)>>(),
             ...);
        },
        tuple);
}

template <class Exception>
void test_Exception_what() {
    Exception exception1("exception");
    ASSERT_EQ(exception1.what(), "exception\n");
    Exception exception2("exception", "file", 1, "func_name");
    ASSERT_EQ(exception2.what(), "exception\n\tfunc_name: file<1>\n");
    Exception exception3(exception2);
    ASSERT_EQ(exception3.what(), "exception\n\tfunc_name: file<1>\n");
    Exception exception4(std::move(exception3));
    ASSERT_EQ(exception4.what(), "exception\n\tfunc_name: file<1>\n");
    auto exception5 = exception4;
    ASSERT_EQ(exception5.what(), "exception\n\tfunc_name: file<1>\n");
    auto exception6 = std::move(exception5);
    ASSERT_EQ(exception6.what(), "exception\n\tfunc_name: file<1>\n");
}

TEST(Exception, what) {
    auto tuple = Exception_Tuple();
    std::apply(
        [](auto &&...args) {
            (test_Exception_what<std::remove_cvref_t<decltype(args)>>(), ...);
        },
        tuple);
}

template <class Exception>
void test_Exception_add_location() {
    Exception exception1("exception");
    ASSERT_EQ(exception1.what(), "exception\n");
    exception1.add_location("file", 1, "func_name");
    ASSERT_EQ(exception1.what(), "exception\n\tfunc_name: file<1>\n");
    Exception exception2(exception1);
    ASSERT_EQ(exception2.what(), "exception\n\tfunc_name: file<1>\n");
    exception2.add_location("file", 2, "func_name");
    ASSERT_EQ(exception2.what(),
              "exception\n\tfunc_name: file<1>\n\tfunc_name: file<2>\n");
}

TEST(Exception, add_location) {
    auto tuple = Exception_Tuple();
    std::apply(
        [](auto &&...args) {
            (test_Exception_add_location<std::remove_cvref_t<decltype(args)>>(),
             ...);
        },
        tuple);
}
#endif // TEST_EXCEPTION
