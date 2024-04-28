#include <iostream>
#include "llframe.hpp"
void throw_exception() {
    __LLFRAME_THROW_EXCEPTION__(llframe::Unknown);
}

void func() {
    __LLFRAME_TRY_CATCH_BEGIN__
    throw_exception();
    __LLFRAME_TRY_CATCH_END__
}
int main() {
    __LLFRAME_TRY_CATCH_BEGIN__
    func();
    __LLFRAME_TRY_END__
    catch (llframe::Exception &e) {
        std::cout << e.what() << std::endl;
    }
}