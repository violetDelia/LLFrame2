#include "test_config.hpp"
#ifdef TEST_EXCEPTION
#include <gtest/gtest.h>
#include "test_common.hpp"
#include <string>

#define DEFAULT_MESSAGE "exception!"
#define DEFAULT_LOCATION "\n"
#define CONCAT_STRING(arg1, arg2)                                              \
    std::string(std::string(arg1) + std::string(arg2))
#define DEFAULT_WAHT CONCAT_STRING(DEFAULT_MESSAGE, DEFAULT_LOCATION)
#define MAKE_LOCATION(file, line, func_name)                                   \
    CONCAT_STRING(                                                             \
        CONCAT_STRING(                                                         \
            CONCAT_STRING(                                                     \
                CONCAT_STRING(                                                 \
                    CONCAT_STRING(CONCAT_STRING("\t", func_name), ": "),       \
                    file),                                                     \
                "<"),                                                          \
            std::to_string(line)),                                             \
        ">\n")
#define MAKE_WAHT_PREFIX(message) CONCAT_STRING(message, DEFAULT_LOCATION)
#define MAKE_WAHT(message, file, line, func_name)                              \
    CONCAT_STRING(MAKE_WAHT_PREFIX(message),                                   \
                  MAKE_LOCATION(file, line, func_name))

template <class Exception>
void test_Exception_default_construct() {
    Exception exception;
    ASSERT_EQ(exception.what(), DEFAULT_WAHT);
}
template <class Exception>
void test_Exception_construct_message(const char *message) {
    Exception exception(message);
    ASSERT_EQ(exception.what(), MAKE_WAHT_PREFIX(message));
}

template <class Exception>
void test_Exception_construct_message_file_line_func_name(
    const char *message, const char *file, const size_t line,
    const char *func_name) {
    Exception exception(message, file, line, func_name);
    ASSERT_EQ(exception.what(), MAKE_WAHT(message, file, line, func_name));
}

template <class Exception>
void test_Exception_construct_file_line_func_name(const char *file,
                                                  const size_t line,
                                                  const char *func_name) {
    Exception exception(file, line, func_name);
    ASSERT_EQ(exception.what(),
              MAKE_WAHT(DEFAULT_MESSAGE, file, line, func_name));
}

template <class Exception>
void test_Exception_construct_copy(const char *message, const char *file,
                                   const size_t line, const char *func_name) {
    Exception exception(message, file, line, func_name);
    auto exception_copy(exception);
    ASSERT_EQ(exception.what(), exception_copy.what());
}

template <class Exception>
void test_Exception_construct_move(const char *message, const char *file,
                                   const size_t line, const char *func_name) {
    Exception exception(message, file, line, func_name);
    auto exception_move(std::move(exception));
    ASSERT_EQ(exception.what(), "");
    ASSERT_EQ(exception_move.what(), MAKE_WAHT(message, file, line, func_name));
}

template <class Exception>
void test_Exception_construct(const char *message, const char *file,
                              const size_t line, const char *func_name) {
    test_Exception_default_construct<Exception>();
    test_Exception_construct_message<Exception>(message);
    test_Exception_construct_file_line_func_name<Exception>(file, line,
                                                            func_name);
    test_Exception_construct_message_file_line_func_name<Exception>(
        message, file, line, func_name);
    test_Exception_construct_copy<Exception>(message, file, line, func_name);
    test_Exception_construct_move<Exception>(message, file, line, func_name);
}

TEST(Exception, construct) {
    APPLY_TUPLE(Exception_Tuple, test_Exception_construct, "", "", 0, "");
    APPLY_TUPLE(Exception_Tuple, test_Exception_construct, "\n", "**@@#", 0,
                ")__!+");
    APPLY_TUPLE(Exception_Tuple, test_Exception_construct, "\t", "||!@\\23", 0,
                ")(*(!@*))");
    APPLY_TUPLE(Exception_Tuple, test_Exception_construct, "???#?@?", "", 0,
                "");
}

template <class Exception>
void test_Exception_what(const char *message, const char *file, size_t line,
                         const char *func_name) {
    Exception exception(message, file, line, func_name);
    ASSERT_EQ(exception.what(), MAKE_WAHT(message, file, line, func_name));
}

TEST(Exception, what) {
    APPLY_TUPLE(Exception_Tuple, test_Exception_what, "", "", 0, "");
    APPLY_TUPLE(Exception_Tuple, test_Exception_what, "\n", "**@@#", 0,
                ")__!+");
    APPLY_TUPLE(Exception_Tuple, test_Exception_what, "\t", "||!@\\23", 0,
                ")(*(!@*))");
    APPLY_TUPLE(Exception_Tuple, test_Exception_what, "???#?@?", "", 0, "");
}

template <class Exception>
void test_Exception_add_location(const char *message, const char *file,
                                 size_t line, const char *func_name) {
    Exception exception(message);
    ASSERT_EQ(exception.what(), MAKE_WAHT_PREFIX(message));
    exception.add_location(file, line, func_name);
    ASSERT_EQ(exception.what(), MAKE_WAHT(message, file, line, func_name));
}

TEST(Exception, add_location) {
    APPLY_TUPLE(Exception_Tuple, test_Exception_add_location, "", "", 0, "");
    APPLY_TUPLE(Exception_Tuple, test_Exception_add_location, "\n", "**@@#", 0,
                ")__!+");
    APPLY_TUPLE(Exception_Tuple, test_Exception_add_location, "\t", "||!@\\23",
                0, ")(*(!@*))");
    APPLY_TUPLE(Exception_Tuple, test_Exception_add_location, "???#?@?", "", 0,
                "");
}
#endif // TEST_EXCEPTION
