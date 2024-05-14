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
        // llframe::memory::Memory<A, llframe::device::GPU> memory(2,0);
        // decltype(tensor) tensor_copy(tensor);

        // for (int i = 0; i < memory.size(); i++) { memory.set(i, i); }

        // auto memory_other = tensor.memory();
        // for (int i = 0; i < memory_other.size(); i++) {
        //     QUICK_PRINT(memory_other.get(i));
        // }

        // memory.fill(5, 5, 3);
        // cudaMemcpy(&a, memory.data(), 4, cudaMemcpyDeviceToHost);
        // for (int i = 0; i < 10; i++) {
        //     std::cout << memory.get(i) << std::endl;
        // }

        // memory.~Memory();
        // std::cout << "1" << std::endl;

        // memory_base.fill(0, {1, 2, 3, 4, 5});
        // llframe::memory::Memory<int, device::GPU> memory_2(10, 0);
        // memory_2.copy_form(memory_base);
        // for (int i = 0; i < 10; i++) { QUICK_PRINT(memory_2.get(i)); }
    } catch (llframe::exception::Exception &e) {
        std::cout << e.what() << std::endl;
    } catch (std::exception &e) { std::cout << e.what() << std::endl; } catch (...) {
        std::cout << "other" << std::endl;
    }
}