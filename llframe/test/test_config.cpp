#include "test_config.hpp"
#ifdef TEST_CONFIG
#include <gtest/gtest.h>
#include "llframe.hpp"

TEST(gtest, test) {
    ASSERT_EQ(1, 1);
    ASSERT_FALSE(false);
}
#endif // TEST_CONFIG