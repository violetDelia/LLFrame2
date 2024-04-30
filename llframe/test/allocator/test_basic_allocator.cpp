#include "test_config.hpp"
#ifdef TEST_ALLOCATOR
#include <gtest/gtest.h>
#include "test_common.hpp"
#include "allocator/basic_allocator.hpp"

template <class Ty>
void test_Basic_Allocator() {
    using allocator = llframe::allocator::Biasc_Allocator<Ty>;
    test_allocator_traits<allocator>();
    if (sizeof(Ty) != 1) {
        ASSERT_THROW(auto placehold = allocator::allocate(
                         std::numeric_limits<size_t>::max()),
                     llframe::exception::Bad_Alloc);
    };
    ASSERT_EQ(allocator::allocate(0), nullptr);
    auto *address = allocator::allocate(5);
    allocator::deallocate(address, 5);
    auto *void_adress = allocator::allocate_bytes(5);
    allocator::deallocate_bytes(void_adress, 5);
};

TEST(Basic_Allocator, all) {
    APPLY_TUPLE(Type_Tuple, test_Basic_Allocator);
};

#endif // TEST_ALLOCATOR