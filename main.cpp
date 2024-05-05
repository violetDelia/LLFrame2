#include <iostream>
#include "llframe.hpp"
#include <array>
#include <cuda_runtime.h>
using namespace llframe;
#define QUICK_PRINT(val) std::cout << val << std::endl;
class A {
public:
    A() {
        count++;
        std::cout << "A:" << count << std::endl;
    }
    ~A() {
        count--;
        std::cout << "~A:" << count << std::endl;
    }
    static inline int count = 0;
};

int main() {
    try {
        llframe::memory::Memory<A, device::CPU> memory_base(10, 0);
        llframe::memory::Memory<A, device::CPU> memory_2(10, 0);
        memory_2.copy_form(memory_base);
        for (int i = 0; i < 10; i++) {}
    } catch (llframe::exception::CUDA_Error &e) {
        std::cout << "cudaerror_t = " << e.what() << std::endl;
    } catch (llframe::exception::Exception &e) {
        std::cout << "llframe:: " << e.what() << std::endl;
    } catch (...) { std::cout << "other" << std::endl; }
}