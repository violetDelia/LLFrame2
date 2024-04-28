#include "test_config.hpp"
#ifdef TEST_BASE_TYPE
#include <gtest/gtest.h>
#include "test_common.hpp"

TEST(base_type, int8_t) {
    ASSERT_EQ(sizeof(llframe::int8_t), 1);
    ASSERT_EQ(std::numeric_limits<llframe::int8_t>::max(), 127);
    ASSERT_EQ(std::numeric_limits<llframe::int8_t>::min(), -128);
    ASSERT_EQ(std::is_integral<llframe::int8_t>::value, true);
    ASSERT_EQ(std::is_floating_point<llframe::int8_t>::value, false);
    ASSERT_EQ(std::is_signed<llframe::int8_t>::value, true);
    ASSERT_EQ(std::is_unsigned<llframe::int8_t>::value, false);
    ASSERT_EQ(std::numeric_limits<llframe::int8_t>::is_bounded, true);
}
TEST(base_type, int16_t) {
    ASSERT_EQ(sizeof(llframe::int16_t), 2);
    ASSERT_EQ(std::numeric_limits<llframe::int16_t>::max(), 32767);
    ASSERT_EQ(std::numeric_limits<llframe::int16_t>::min(), -32768);
    ASSERT_EQ(std::is_integral<llframe::int16_t>::value, true);
    ASSERT_EQ(std::is_floating_point<llframe::int16_t>::value, false);
    ASSERT_EQ(std::is_signed<llframe::int16_t>::value, true);
    ASSERT_EQ(std::is_unsigned<llframe::int16_t>::value, false);
    ASSERT_EQ(std::numeric_limits<llframe::int16_t>::is_bounded, true);
}
TEST(base_type, int32_t) {
    ASSERT_EQ(sizeof(llframe::int32_t), 4);
    ASSERT_EQ(std::numeric_limits<llframe::int32_t>::max(), 2147483647);
    ASSERT_EQ(std::numeric_limits<llframe::int32_t>::min(), -2147483648);
    ASSERT_EQ(std::is_integral<llframe::int32_t>::value, true);
    ASSERT_EQ(std::is_floating_point<llframe::int32_t>::value, false);
    ASSERT_EQ(std::is_signed<llframe::int32_t>::value, true);
    ASSERT_EQ(std::is_unsigned<llframe::int32_t>::value, false);
    ASSERT_EQ(std::numeric_limits<llframe::int32_t>::is_bounded, true);
}
TEST(base_type, int64_t) {
    ASSERT_EQ(sizeof(llframe::int64_t), 8);
    ASSERT_EQ(std::numeric_limits<llframe::int64_t>::max(), LLONG_MAX);
    ASSERT_EQ(std::numeric_limits<llframe::int64_t>::min(), LLONG_MIN);
    ASSERT_EQ(std::is_integral<llframe::int64_t>::value, true);
    ASSERT_EQ(std::is_floating_point<llframe::int64_t>::value, false);
    ASSERT_EQ(std::is_signed<llframe::int64_t>::value, true);
    ASSERT_EQ(std::is_unsigned<llframe::int32_t>::value, false);
    ASSERT_EQ(std::numeric_limits<llframe::int64_t>::is_bounded, true);
}
TEST(base_type, uint8_t) {
    ASSERT_EQ(sizeof(llframe::uint8_t), 1);
    ASSERT_EQ(std::numeric_limits<llframe::uint8_t>::max(), 255);
    ASSERT_EQ(std::numeric_limits<llframe::uint8_t>::min(), 0);
    ASSERT_EQ(std::is_integral<llframe::uint8_t>::value, true);
    ASSERT_EQ(std::is_floating_point<llframe::uint8_t>::value, false);
    ASSERT_EQ(std::is_signed<llframe::uint8_t>::value, false);
    ASSERT_EQ(std::is_unsigned<llframe::uint8_t>::value, true);
    ASSERT_EQ(std::numeric_limits<llframe::uint8_t>::is_bounded, true);
}
TEST(base_type, uint16_t) {
    ASSERT_EQ(sizeof(llframe::uint16_t), 2);
    ASSERT_EQ(std::numeric_limits<llframe::uint16_t>::max(), 65535);
    ASSERT_EQ(std::numeric_limits<llframe::uint16_t>::min(), 0);
    ASSERT_EQ(std::is_integral<llframe::uint16_t>::value, true);
    ASSERT_EQ(std::is_floating_point<llframe::uint16_t>::value, false);
    ASSERT_EQ(std::is_signed<llframe::uint16_t>::value, false);
    ASSERT_EQ(std::is_unsigned<llframe::uint16_t>::value, true);
    ASSERT_EQ(std::numeric_limits<llframe::uint16_t>::is_bounded, true);
}
TEST(base_type, uint32_t) {
    ASSERT_EQ(sizeof(llframe::uint32_t), 4);
    ASSERT_EQ(std::numeric_limits<llframe::uint32_t>::max(), 4294967295);
    ASSERT_EQ(std::numeric_limits<llframe::uint32_t>::min(), 0);
    ASSERT_EQ(std::is_integral<llframe::uint32_t>::value, true);
    ASSERT_EQ(std::is_floating_point<llframe::uint32_t>::value, false);
    ASSERT_EQ(std::is_signed<llframe::uint32_t>::value, false);
    ASSERT_EQ(std::is_unsigned<llframe::uint8_t>::value, true);
    ASSERT_EQ(std::numeric_limits<llframe::uint32_t>::is_bounded, true);
}
TEST(base_type, uint64_t) {
    ASSERT_EQ(sizeof(llframe::uint64_t), 8);
    ASSERT_EQ(std::numeric_limits<llframe::uint64_t>::max(), ULLONG_MAX);
    ASSERT_EQ(std::numeric_limits<llframe::uint64_t>::min(), 0);
    ASSERT_EQ(std::is_integral<llframe::uint64_t>::value, true);
    ASSERT_EQ(std::is_floating_point<llframe::uint64_t>::value, false);
    ASSERT_EQ(std::is_signed<llframe::uint64_t>::value, false);
    ASSERT_EQ(std::is_unsigned<llframe::uint64_t>::value, true);
    ASSERT_EQ(std::numeric_limits<llframe::uint64_t>::is_bounded, true);
}
TEST(base_type, float32_t) {
    ASSERT_EQ(sizeof(llframe::float32_t), 4);
    ASSERT_EQ(std::numeric_limits<llframe::float32_t>::max(), FLT_MAX);
    ASSERT_EQ(std::numeric_limits<llframe::float32_t>::min(), FLT_MIN);
    ASSERT_EQ(std::is_integral<llframe::float32_t>::value, false);
    ASSERT_EQ(std::is_floating_point<llframe::float32_t>::value, true);
    ASSERT_EQ(std::is_signed<llframe::float32_t>::value, true);
    ASSERT_EQ(std::is_unsigned<llframe::float32_t>::value, false);
    ASSERT_EQ(std::numeric_limits<llframe::float32_t>::is_bounded, true);
    ASSERT_EQ(std::numeric_limits<llframe::float32_t>::has_infinity, true);
    ASSERT_EQ(std::numeric_limits<llframe::float32_t>::has_quiet_NaN, true);
}
TEST(base_type, float64_t) {
    ASSERT_EQ(sizeof(llframe::float64_t), 8);
    ASSERT_EQ(std::numeric_limits<llframe::float64_t>::max(), DBL_MAX);
    ASSERT_EQ(std::numeric_limits<llframe::float64_t>::min(), DBL_MIN);
    ASSERT_EQ(std::is_integral<llframe::float64_t>::value, false);
    ASSERT_EQ(std::is_floating_point<llframe::float64_t>::value, true);
    ASSERT_EQ(std::is_signed<llframe::float64_t>::value, true);
    ASSERT_EQ(std::is_unsigned<llframe::float64_t>::value, false);
    ASSERT_EQ(std::numeric_limits<llframe::float64_t>::is_bounded, true);
    ASSERT_EQ(std::numeric_limits<llframe::float64_t>::has_infinity, true);
    ASSERT_EQ(std::numeric_limits<llframe::float64_t>::has_quiet_NaN, true);
}
TEST(base_type, size_t) {
    ASSERT_EQ(std::is_integral<llframe::size_t>::value, true);
    ASSERT_EQ(std::is_floating_point<llframe::size_t>::value, false);
    ASSERT_EQ(std::is_signed<llframe::size_t>::value, false);
    ASSERT_EQ(std::is_unsigned<llframe::size_t>::value, true);
    ASSERT_EQ(std::numeric_limits<llframe::size_t>::is_bounded, true);
}
TEST(base_type, ptrdiff_t) {
    ASSERT_EQ(std::is_integral<llframe::ptrdiff_t>::value, true);
    ASSERT_EQ(std::is_floating_point<llframe::ptrdiff_t>::value, false);
    ASSERT_EQ(std::is_signed<llframe::ptrdiff_t>::value, true);
    ASSERT_EQ(std::is_unsigned<llframe::ptrdiff_t>::value, false);
    ASSERT_EQ(std::numeric_limits<llframe::ptrdiff_t>::is_bounded, true);
}
#endif // TEST_BASE_TYPE
