#include <iostream>
#include "llframe.hpp"
#include <array>
#include <cuda_runtime.h>
#include <cmath>
using namespace llframe;
#define QUICK_PRINT(val) std::cout << val << std::endl;

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
int main() {
    try {
        llframe::memory::Memory<A, llframe::device::CPU> memory(2,0);
        auto c(memory);
        
        
    } catch (llframe::exception::Exception &e) {
        std::cout << e.what() << std::endl;
    } catch (std::exception &e) { std::cout << e.what() << std::endl; } catch (...) {
        std::cout << "other" << std::endl;
    }
}